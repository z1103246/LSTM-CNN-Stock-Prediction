import os

import numpy as np
import pandas as pd

from tensorflow.keras.utils import Sequence

from ProcessData import generate_chart
from ProcessData import generate_timeserie
from ProcessData import get_indicators
from ProcessData import log_return
from ProcessData import check_indicators


class StockDataGenerator(Sequence):
    def __init__(self, df, etf, chart_indicators, ts_indicators, look_forward, look_back, batch_size, single_output=True):
        assert batch_size > look_back, "Batch size must be greater than the look_back window"
        assert look_forward < batch_size, "Look forward must be smaller than batch size."
        assert check_indicators(df, (chart_indicators + ts_indicators)
                                ), "Make sure all neccessary technical indicators have been computed and are in the dataframe."

        self.df = df
        self.etf = etf
        self.ts_indicators = ts_indicators
        self.chart_indicators = chart_indicators
        self.look_forward = look_forward
        self.look_back = look_back
        self.batch_size = batch_size
        self.single_output = single_output

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size

        if batch_start < self.look_back:
            batch_start = self.look_back

        if batch_end >= len(self.df):
            batch_end = len(self.df) - self.look_forward

        # The data on index batch_end is not included
        charts = np.array(generate_chart(
            self.df, self.etf, self.look_back, batch_start, batch_end, self.chart_indicators))
        ts = np.array(generate_timeserie(self.df, self.look_back,
                                         batch_start, batch_end, self.ts_indicators))

        # if not self.single_output:
        #     logreturn_pred = [np.log10(self.df.loc[batch_end + i, "close"] /
        #                                self.df.loc[batch_end - 1, "close"]) for i in range(self.look_forward)]
        # else:
        #     logreturn_pred = [np.log10(
        #         self.df.loc[batch_end + self.look_forward - 1, "close"]/self.df.loc[batch_end - 1, "close"])]
        logreturn_pred = []
        if self.single_output:
            for i in range(batch_start, batch_end):
                logreturn_pred.append([np.log10(self.df.loc[i + self.look_forward, "close"] /
                                                self.df.loc[i - 1, "close"])] * 3)
        else:
            for i in range(batch_start, batch_end):
                logreturn_pred.append([np.log10(self.df.loc[i + j, "close"] /
                                                self.df.loc[i - 1, "close"]) for j in range(self.look_forward)] * 3)

        print(np.array(logreturn_pred).shape)
        print(ts.shape)

        return [charts, ts], np.array(logreturn_pred)
