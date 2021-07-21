from typing import List, Tuple
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class BaseRenderer:
    def show(self):
        plt.show()

    def input_time_series(self, data: pd.DataFrame, features: List[str], asset_index_name: str):
        assets = data.index.get_level_values(asset_index_name).unique()
        data = data.unstack(level=asset_index_name)

        fig, axes = plt.subplots(len(features), 1, figsize=(15, 10))
        for feature_ax_idx, feature in enumerate(features):
            axes[feature_ax_idx].set_ylabel(feature)
            axes[feature_ax_idx].set_xlabel('Time')

            for asset in assets:
                plotting_data = data[(feature, asset)]

                plotting_data.plot(ax=axes[feature_ax_idx])

        axes[0].legend()

    def portfolio_value_time_series(self, data: List[Tuple[datetime, float]]):
        fig, ax = plt.subplots()

        datetimes, values = list(zip(*data))
        ax.plot(datetimes, values)

        fmt_half_year = mdates.DayLocator(interval=10)
        ax.xaxis.set_major_locator(fmt_half_year)

        fmt_month = mdates.DayLocator(interval=5)
        ax.xaxis.set_minor_locator(fmt_month)

        # Text in the x axis will be displayed in 'YYYY-mm' format.
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Round to nearest years.
        datemin = np.datetime64(datetimes[0], 'h')
        datemax = np.datetime64(datetimes[-1], 'h') + np.timedelta64(1, 'h')
        ax.set_xlim(datemin, datemax)

        # Format the coords message box, i.e. the numbers displayed as the cursor moves
        # across the axes within the interactive GUI.
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ax.format_ydata = lambda x: f'${x:.2f}'  # Format the price.
        ax.grid(True)

        # Rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them.
        fig.autofmt_xdate()
