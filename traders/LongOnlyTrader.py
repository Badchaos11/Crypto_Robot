from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

import logging

from secrets import client


class LongOnlyTrader:

    def __init__(self, symbol, bar_length, return_thresh, volume_thresh, units, position=0):
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d",
                                    "1w", "1M"]
        self.position = position
        self.units = units

        self.trades = 0
        self.trade_values = []

        self.twm = ThreadedWebsocketManager()

        self.return_thresh = return_thresh
        self.volume_thresh = volume_thresh

    def start_trading(self, historical_days):
        self.twm.start()

        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol=self.symbol, interval=self.bar_length,
                                 days=historical_days)
            self.twm.start_kline_socket(callback=self.stream_candles,
                                        symbol=self.symbol, interval=self.bar_length)
            self.twm.join()

    def get_most_recent(self, symbol, interval, days):

        now = datetime.utcnow()
        past = str(now - timedelta(days=days))

        bars = client.get_historical_klines(symbol=symbol, interval=interval,
                                            start_str=past, end_str=None, limit=1000)
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:, 0], unit="ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace=True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["Complete"] = [True for row in range(len(df) - 1)] + [False]

        self.data = df

    def stream_candles(self, msg):

        event_time = pd.to_datetime(msg["E"], unit="ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit="ms")
        first = float(msg["k"]["o"])
        high = float(msg["k"]["h"])
        low = float(msg["k"]["l"])
        close = float(msg["k"]["c"])
        volume = float(msg["k"]["v"])
        complete = msg["k"]["x"]
        print(f"Time {start_time} | Close {close} | Complete {complete} | Position {self.position}")
        #print(".", end="", flush=True)

        self.data.loc[start_time] = [first, high, low, close, volume, complete]

        if complete:
            self.define_strategy()
            self.execute_trades()

    def define_strategy(self):

        df = self.data.copy()

        df = df[['Close', 'Volume']].copy()
        df['returns'] = np.log(df.Close / df.Close.shift())
        df['vol_ch'] = np.log(df.Volume.div(df.Volume.shift(1)))
        df.loc[df.vol_ch > 3, 'vol_ch'] = np.nan
        df.loc[df.vol_ch < -3, 'vol_ch'] = np.nan

        cond1 = df.returns >= self.return_thresh
        cond2 = df.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])

        df['position'] = 1
        df.loc[cond1 & cond2, 'position'] = 0

        self.prepare_data = df.copy()

    def execute_trades(self):
        if self.prepare_data['position'].iloc[-1] == 1:
            if self.position == 0:
                order = client.create_order(symbol=self.symbol, side='BUY', type='MARKET', quantity=self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepare_data['position'].iloc[-1] == 0:
            if self.position == 1:
                order = client.create_order(symbol=self.symbol, side='SELL', type='MARKET', quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0

    def report_trade(self, order, going):

        # extract data from order object
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit="ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)

        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units)

        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3)
            self.cum_profits = round(np.sum(self.trade_values), 3)
        else:
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)

        # print trade report
        print(2 * "\n" + 100 * "-")
        print("{} | {}".format(time, going))
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(time, real_profit, self.cum_profits))
        print(f"Trade Value for last trade: {order['cummulativeQuoteQty']}")
        print(f"Returns {self.prepare_data['returns'].iloc[-1]}")
        print(f"Volume change {self.prepare_data['vol_ch'].iloc[-1]}")
        print(f"Position {self.prepare_data['position'].iloc[-1]}")
        print(100 * "-" + "\n")


if __name__ == '__main__':
    symbol = "BTCUSDT"
    bar_length = "1m"
    return_thresh = 0
    volume_thresh = [-3, 3]
    units = 0.001
    position = 0

    trader = LongOnlyTrader(symbol=symbol, bar_length=bar_length, return_thresh=return_thresh,
                            volume_thresh=volume_thresh, units=units, position=position)

    trader.start_trading(historical_days=1 / 24)
