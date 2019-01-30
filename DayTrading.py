"""
df = pd.read_csv(
    'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AMD&datatype=csv&interval=1min&outputsize=full&apikey=YJRUTXHBWJOD0MAY',
    index_col='timestamp', parse_dates=True)

1. day trading: classify AMD as buy (+1) or sell (-1) or hold (0)
    - based on the past 20 three-min candles [past 60 minutes]
    - candle datapoints:
        - open, close, high, low, volume from 3 one minute candles
    - supervised on sharpe ratio [risk-adj returns] after holding for 5 candles (15 min)
        - more than 1: buy
    - dataset size: about 2000 samples (looking at every 3 min candle for past week)
    - only run this during the day, gaps on open will only confuse the learner
        [9:30...10:30] to predict -> 10:45
        [9:31...10:31] to predict -> 10:46
                          ...
        [2:44...3:44]  to predict ->  3:59
        [2:45...3:45]  to predict ->  4:00
"""
