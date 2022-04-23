from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from secrets import client


class LongShortSMATrader:

    def __init__(self, symbol, bar_length, sma_s, sma_m, sma_l, units, position=0):

        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d",
                                    "1w", "1M"]
        self.units = units
        self.position = position
        self.trades = 0
        self.trade_values = []

        # *****************add strategy-specific attributes here******************
        self.SMA_S = sma_s
        self.SMA_M = sma_m
        self.SMA_L = sma_l
        # ************************************************************************

    def start_trading(self, historical_days):

        self.twm = ThreadedWebsocketManager()
        self.twm.start()

        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol=self.symbol, interval=self.bar_length,
                                 days=historical_days)
            self.twm.start_kline_socket(callback=self.stream_candles,
                                        symbol=self.symbol, interval=self.bar_length)
            self.twm.join()
        # "else" to be added later in the course

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

        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit="ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit="ms")
        first = float(msg["k"]["o"])
        high = float(msg["k"]["h"])
        low = float(msg["k"]["l"])
        close = float(msg["k"]["c"])
        volume = float(msg["k"]["v"])
        complete = msg["k"]["x"]
        print(f"Time {start_time} | Close {close} | Complete {complete} | Position {self.position}")

        # stop trading session
        if self.trades >= 5:  # stop stream after 5 trades
            self.twm.stop()
            if self.position == 1:
                order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
            elif self.position == -1:
                order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
            else:
                print("STOP")

        # print out
        #print(".", end="", flush=True)  # just print something to get a feedback (everything OK)

        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]

        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.define_strategy()
            self.execute_trades()

    def define_strategy(self):

        data = self.data.copy()

        # ******************** define your strategy here ************************
        data = data[["Close"]].copy()

        data["SMA_S"] = data.Close.rolling(window=self.SMA_S).mean()
        data["SMA_M"] = data.Close.rolling(window=self.SMA_M).mean()
        data["SMA_L"] = data.Close.rolling(window=self.SMA_L).mean()

        data.dropna(inplace=True)

        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)

        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1
        # ***********************************************************************

        self.prepared_data = data.copy()

    def execute_trades(self):
        if self.prepared_data["position"].iloc[-1] == 1:  # if position is long -> go/stay long
            if self.position == 0:
                order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL")
                time.sleep(1)
                order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0:  # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == -1:
                order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
        if self.prepared_data["position"].iloc[-1] == -1:  # if position is short -> go/stay short
            if self.position == 0:
                order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL")
                time.sleep(1)
                order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING SHORT")
            self.position = -1

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
        print(f"SMAS {self.prepared_data['SMA_S'].iloc[-1]}")
        print(f"SMAL  {self.prepared_data['SMA_L'].iloc[-1]}")
        print(f"SMAM {self.prepared_data['SMA_M'].iloc[-1]}")
        print(100 * "-" + "\n")


if __name__ == '__main__':
    symbol = "BTCUSDT"
    bar_length = "1m"
    sma_s = 15
    sma_m = 50
    sma_l = 200
    units = 0.001
    position = 0

    trader = LongShortSMATrader(symbol=symbol, bar_length=bar_length, sma_s=sma_s, sma_m=sma_m, sma_l=sma_l,
                                units=units, position=position)

    trader.start_trading(historical_days=4/24)
