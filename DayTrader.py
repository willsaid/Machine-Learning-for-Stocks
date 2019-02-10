import Trader
import Candle

import pandas as pd
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier  # multi layer neural network
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

"""
df = pd.read_csv(
    'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AMD&datatype=csv&interval=1min&outputsize=full&apikey=YJRUTXHBWJOD0MAY',
    index_col='timestamp', parse_dates=True)

1. day trading: classify AMD as buy (+1) or short (-1) or hold (0) for next 15 min
    - based on the past 20 three-min candles [past 60 minutes]
    - candle datapoints:
        - open, close, high, low, volume from three one minute candles merged
    - supervised on sharpe ratio [risk-adj returns] between now and after holding for 5 candles (15 min)
        - more than 1: buy
        - less than 0: short
        - else hold
    - dataset size: about 2000 samples (looking at every 3 min candle for past week)
        ex: [9:30...10:30] to predict -> 10:45
"""
class DayTrader(Trader.Trader):

    def __init__(self, df=None):
        self.tradingType = 'DayTrader'
        if df is None:
            self.df = pd.read_csv('amd.csv', index_col='timestamp', parse_dates=True)
        else:
            # print df
            self.df = df
        x, y = self.create_dataset()
        self.train_xs, self.test_xs, self.train_ys, self.test_ys = self.split_training_testing_sets(x, y)

    def create_dataset(self):
        """
        returns tuple (X, y) of the form:

        # last 20 candles
        >>> X = [
                  [ 1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000,
                    1796.199951,1799.939941,1791.829956,1799.839966,1799.839966,3312160000,
                    1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000,
                    1796.199951,1799.939941,1791.829956,1799.839966,1799.839966,3312160000,
                    1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000,
                    ... x 20
                                                                                          ]
                    .
                    .
                    .
                ]

        # sharpe ratio over next 15 min is: BUY>1, SHORT<1, else HOLD
        >>> y = [1.0, -1.0, . . .]

        """
        xs, ys, gains = [], [], []

        # df -> one min candles. [4:-1] is on purpose.
        one_min_candles = []
        for index, row in self.df[4:-1].iterrows():
            open, high, low, close, volume = row.values
            one_min_candles.append(Candle.Candle(open, high, low, close, volume))

        # one min candles -> 3  min candles
        three_min_candles = []
        for one_min_candle_index in range(0, len(one_min_candles) - 2, 3):
            three_candle = Candle.Candle.initFromCandles(one_min_candles[one_min_candle_index], one_min_candles[one_min_candle_index + 1], one_min_candles[one_min_candle_index + 2])
            three_min_candles.append(three_candle)

        # find corresponding sharpe (Ys):
        # use last 20 candles [current, as well as past 19)
        # to find sharpe between now and after next 5 candles
        for index in range(19, len(three_min_candles) - 5):
            pastCandleValues = []
            for candle in three_min_candles[index-19:index+1]:
                # pastCandleValues.append(candle.open)
                pastCandleValues.append(candle.close)
                # pastCandleValues.append(candle.high)
                # pastCandleValues.append(candle.low)
            xs.append(pastCandleValues)
            next_candles = three_min_candles[index+1:index+6]
            gains.append((next_candles[-1].close - next_candles[1].open) / next_candles[1].open - 1) # normalized, like -.01 1% loss
            ys.append(self.classifySharpe(self.sharpe(next_candles)))

        # used for predicting tomorrow
        self.current_sample = xs[-1]

        splitting_index = int(-1 * (1.0 - self.training_split) * len(three_min_candles))
        self.testingX = xs[:splitting_index]
        self.testingY = ys[:splitting_index]
        self.testingGains = gains[:splitting_index]
        self.trainingGains = gains[splitting_index:]
        xs = xs[splitting_index:]
        ys = ys[splitting_index:]

        # shuffle them around: this somehow results in a super high decision tree accuracy of over 70% , but it should be random right?
        import random
        zipped = list(zip(xs, ys))
        random.shuffle(zipped)
        return zip(*zipped)

    def classifySharpe(self, sharpe):

        return 1.0 if sharpe > 1.0 else -1 # simple version, seems to do better

        # if sharpe < 0: return -1 # short
        # if sharpe > 1: return 1  # buy
        # return 0                 # hold

    def sharpe(self, candles):
        """
        candles: [3 min candles], length is 5
        """
        annual_samples = 252 * 6.5 * 20
        annual_risk_free = 1.025 # 2.5 percent annual t-bill
        risk_free = (annual_risk_free ** (1. / annual_samples)) - 1
        returns = candles[-1].close - candles[0].open
        mean = np.mean(returns - risk_free)
        stddev = np.std([candle.close for candle in candles])
        return mean / stddev
