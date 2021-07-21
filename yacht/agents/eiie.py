import logging

import torch

from exceptions import EndOfDataStreamError
from .base import TrainBaseAgent, BackTestBaseAgent
from agents.strategies import EIIENetwork
from config import Config
from environment.environment import Environment


logger = logging.getLogger(__file__)


class EIIEMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_transaction_remainder_factor(self, new_w, previous_w, k: int = None):
        """

        Args:
            new_w: target portfolio vector, first element is btc
            previous_w: rebalanced last period portfolio vector, first element is btc
            k: if != None the factor will be calculated within 'k' iterations. Usually this is useful
                in training where speed is important.

        Returns:
            Ut = compute transaction remainder factor for normalizing
        """

        commission_rate = self.environment.market.commission
        c_s = commission_rate
        c_p = commission_rate

        iteration = 0
        mu0 = torch.tensor(1).to(device=new_w.device)
        mu1 = torch.tensor(1 - c_s - c_p + c_s * c_p).to(device=new_w.device)
        while (k and iteration < k) or torch.all(torch.abs(mu1 - mu0) > 1e-10):
            mu0 = mu1
            mu1 = (1 - c_p * previous_w[:, 0] -
                   (c_s + c_p - c_s * c_p) *
                   torch.sum(
                       torch.maximum(previous_w[:, 1:] - mu1 * new_w[:, 1:], torch.tensor(0).to(device=new_w.device))
                   )) / \
                  (1 - c_p * new_w[:, 0])

            iteration += 1

        return mu1


class TrainEIIEAgent(TrainBaseAgent):
    def build_strategy(self, environment: Environment, config: Config):
        market = environment.market

        network = EIIENetwork(
            num_features=len(market.features),
            num_assets=len(market.tickers),
            window_size=config.input_config.window_size,
            commission=market.commission
        )
        network = network.to(config.hardware_config.device)

        return network

    def train(self):
        training_config = self.config.training_config

        logger.info('Training...')
        for step in range(self.start_training_step, training_config.steps + 1):
            self.current_step = step
            self.optimizer.zero_grad()

            X, y, previous_w, batch_new_w_datetime = self.get_train_data()

            new_w = self.strategy(X, previous_w)
            self.loss = self.strategy.compute_loss(new_w, y)

            self.loss.backward()
            self.optimizer.step()

            if step % training_config.learning_rate_decay_steps:
                self.scheduler.step()

            # TODO: See if I should normalize the weights before, or not
            new_w = new_w.detach().cpu().numpy()
            self.environment.set_portfolio_weights(batch_new_w_datetime, new_w)

            if step % training_config.validation_every_step == 0 and step != 0:
                self.strategy.train(mode=False)
                with torch.no_grad():
                    X_val, y_val, previous_w_val, batch_new_w_datetime_val = self.get_validation_data()

                    new_w_val = self.strategy(X_val, previous_w_val)
                    metrics = self.strategy.compute_metrics(new_w_val, y_val)
                    self.log_metrics(step, metrics)

                self.strategy.train(mode=True)

            if step % training_config.save_every_step == 0 and step != 0:
                self.save_network(step, self.loss)

        if self.current_step % training_config.save_every_step != 0:
            self.save_network(self.current_step, self.loss)

    def log_metrics(self, step: int, metrics: dict):
        self.log(step, f'Portfolio value: {metrics["portfolio_value"].detach().cpu().numpy()}')
        self.log(step, f'Sharp ratio: {metrics["sharp_ratio"].detach().cpu().numpy()}\n')


class BackTestEIIEAgent(EIIEMixin, BackTestBaseAgent):
    def build_strategy(self, environment: Environment, config: Config):
        market = environment.market

        network = EIIENetwork(
            num_features=len(market.features),
            num_assets=len(market.tickers),
            window_size=config.input_config.window_size,
            commission=market.commission
        )
        network = network.to(config.hardware_config.device)
        network.train(mode=False)

        return network

    def trade(self):
        has_data = True
        while has_data:
            try:
                X, y, previous_w, batch_new_w_datetime = self.get_data()

                with torch.no_grad():
                    new_w = self.strategy(X, previous_w)

                portfolio_gain, normalized_new_w = self.compute_gain(y, previous_w, new_w)
                batch_new_w_datetime = batch_new_w_datetime[0]

                portfolio_gain = portfolio_gain.detach().cpu().numpy()
                normalized_new_w = normalized_new_w.detach().cpu().numpy()
                # TODO: This is not really ok, because if we don't test on all the data is misleading.
                if self.current_step % self.config.training_config.validation_every_step == 0 \
                        and self.current_step != 0:
                    self.log(self.current_step, f'New w: {normalized_new_w}')

                self.portfolio_value = self.portfolio_value * portfolio_gain

                self.portfolio_values.append(
                    (batch_new_w_datetime, self.portfolio_value)
                )
                self.environment.set_portfolio_weights(batch_new_w_datetime, normalized_new_w)

                self.current_step += 1
            except EndOfDataStreamError:
                has_data = False

        # TODO: Think of a better renderer hierarchy.
        self.environment.renderer.portfolio_value_time_series(self.portfolio_values)
        self.environment.renderer.show()

    def compute_gain(self, y, previous_w, new_w):
        # TODO: Shouldn't we +/- the cash weight after sell/buy ?
        transaction_remainder_factor = self.compute_transaction_remainder_factor(new_w, previous_w)

        y = y[0]
        new_w = new_w[0]
        transaction_remainder_factor = transaction_remainder_factor[0]

        future_price = torch.cat([
            torch.ones(size=(1, )).to(device=y.device),
            y[0, :]
        ], dim=0)
        future_w = new_w.dot(future_price)

        portfolio_change = transaction_remainder_factor * future_w
        normalized_new_w = transaction_remainder_factor * (new_w * future_price) / future_w

        return portfolio_change, normalized_new_w
