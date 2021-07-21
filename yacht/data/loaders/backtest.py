from datetime import datetime

from config import InputConfig
from .base import BaseDataLoader
from ..market import BaseMarket
from ..renderers import BaseRenderer


class BackTestDataLoader(BaseDataLoader):
    def __init__(
            self,
            market: BaseMarket,
            renderer: BaseRenderer,
            input_config: InputConfig,
            window_size_offset: int = 1,
    ):
        super().__init__(
            market,
            renderer,
            input_config,
            window_size_offset
        )

        self.step = 0

    def get_batch_size(self) -> int:
        return 1

    def get_first_batch_start_datetime(self) -> datetime:
        batch_start_datetime = self.input_config.start_datetime + self.step * self.input_config.data_frequency.timedelta
        self.step = self.step + 1

        return batch_start_datetime
