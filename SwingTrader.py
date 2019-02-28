import Trader
from Candle import Candle
import pandas as pd

"""
swing trading classifier: predict if market (S&P) will open in GREEN (+1) or RED (-1) tomorrow
"""

class SwingTrader(Trader.Trader):



    """Read in historical market data and create dataset"""
    def __init__(self, past_days=5, excluding_last=252, df=None):
        self.tradingType = 'SwingTrader'
        self.excluding_last = excluding_last
        # s&p 500 (^gspc) 1-day candles from past 5 years
        if df is None:
            self.df = pd.read_csv('^GSPC.csv', index_col='Date', parse_dates=True)
            self.df = self.df[:-1 * excluding_last] # for example, set = 252 for up to 2017
        else:
            # print df
            self.df = df[:-1 * excluding_last]
            # print self.df

        x, y = self.create_dataset(past_days)
        self.train_xs, self.test_xs, self.train_ys, self.test_ys = self.split_training_testing_sets(x, y)

        # self.knn()
        # self.neural_network()
        # self.decision_tree()
        # self.random_forest()

    def last_remaining(self):
        """
        returns [Candle]
        of the last year (if excluding_last = 252 days)
        """
        df = pd.read_csv('^GSPC.csv', index_col='Date', parse_dates=True)
        df = df[-1 * self.excluding_last:] # for example, past 252 days

        # for 2017
        # years_to_chop_off = 10 # 1: chop off 2018 to get just 2017
        # df = df[:-252*years_to_chop_off]

        # print "self.excluding_last", self.excluding_last

        return Candle.swingCandlesFromDataframe(df)

    def create_dataset(self, past_days):
        """
        returns tuple (X, y) of the form:

        >>> X = [
                  [ 1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000
                    1796.199951,1799.939941,1791.829956,1799.839966,1799.839966,3312160000
                    1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000
                    1796.199951,1799.939941,1791.829956,1799.839966,1799.839966,3312160000
                    1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000 ]

                    [1791.030029,1795.979980,1772.880005,1781.560059,1781.560059,4045200000],
                    [1776.010010,1798.030029,1776.010010,1797.020020,1797.020020,3775990000],  # ...
                    [1791.030029,1795.979980,1772.880005,1781.560059,1781.560059,4045200000],  # day before yesterday
                    [1776.010010,1798.030029,1776.010010,1797.020020,1797.020020,3775990000],  # yesterday
                    [1791.030029,1795.979980,1772.880005,1781.560059,1781.560059,4045200000]   # today Open,High,Low,Close,Adj Close,Volume
                    .
                    .
                    .
                ]

        # 1 = next day opens green, -1 = red
        >>> y = [1.0, -1.0, . . .]

        """
        xs, ys = [], []

        for index, row in self.df[past_days - 1 : -1].iterrows():
            # past X days, including today
            pastXDays = self.df[:index].tail(past_days).values.tolist()

            # delete me: just uses past closing days!
            pastXDays = [x[3] for x in pastXDays]
            # print pastXDays
            # print 1/0
            # normalize
            first_val = float(pastXDays[0])
            pastXDays = [x/first_val for x in pastXDays]

            # print pastXDays
            xs.append(pastXDays)
            #
            #
            # xs.append(self.flatsten(pastXDays))
            # print pastXDays
            # print self.flatten(pastXDays)



            # print 1/0
            # just today
            # xs.append(row.values)

            # determine if the following day was red or green
            nextRow = self.df[index:].head(2).tail(1).values # tomorrow
            nextOpen = nextRow[0][0] # OPEN IS FIRST COLUMN!
            isGreen = 1.0 if nextOpen > row['Open'] else -1.0
            ys.append(isGreen)

        # used for predicting tomorrow
        # print self.df
        self.current_sample = xs[-1]
        self.xs = xs
        self.ys = ys

        # shuffle them around: this somehow results in a super high decision tree accuracy of over 70% , but it should be random right?
        import random
        zipped = list(zip(xs, ys))
        random.shuffle(zipped)
        return zip(*zipped)









#
