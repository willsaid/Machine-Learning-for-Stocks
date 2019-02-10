import numpy as np
import pandas as pd

class Candle(object):
    def __init__(self, open, close, high, low, volume):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume

    @staticmethod
    def initFromCandles(candle1, candle2, candle3):
        """ merge 3 one min candles into one big 3 min candle"""
        candles = [candle1, candle2, candle3]
        candle = Candle(0,0,0,0,0)
        candle.open = candle1.open
        candle.close = candle3.open
        candle.high = max([candle.high for candle in candles])
        candle.low = min([candle.low for candle in candles])
        candle.volume = np.mean([candle.volume for candle in candles])
        return candle

    @staticmethod
    def swingCandlesFromDataframe(df):
        """
        returns [Candle]

        currently only supports GSPC, from Yahoo Finance
        """
        # print df
        candles = []
        for index, row in df.iterrows():
            open, high, low, close, adjclose, volume = row.values
            # Open,High,Low,Close,Adj Close,Volume # GSPC!!!
            candles.append(Candle(open,high,low,close,volume))
        return candles

    @staticmethod
    def dayCandlesFromDataframe(df):
        """
        returns [Candle]

        works with Alpha Vantage api, like AMD
        """
        # print df
        candles = []
        for index, row in df.iterrows():
            open, high, low, close, volume = row.values
            # Open,High,Low,Close,Adj Close,Volume # GSPC!!!
            candles.append(Candle(open,high,low,close,volume))
        return candles
