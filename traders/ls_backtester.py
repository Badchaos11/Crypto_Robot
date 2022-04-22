import pandas as pd
import numpy as np
from itertools import product


class Long_Short_Backtester_Futures:
    ''' Class for the vectorized backtesting of simple Long-Short trading strategies.

    Attributes
    ============
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade


    Methods
    =======
    get_data:
        imports the data.

    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).

    prepare_data:
        prepares the data for backtesting.

    run_backtest:
        runs the strategy backtest.

    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold.

    optimize_strategy:
        backtests strategy for different parameter values incl. optimization and reporting (wrapper).

    find_best_strategy:
        finds the optimal strategy (global maximum).


    print_performance:
        calculates and prints various performance metrics.

    '''

    def __init__(self, filepath, symbol, start, end, tc):

        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def __repr__(self):
        return "Long_Short_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)

    def get_data(self):
        ''' Imports the data.
        '''
        raw = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw

    def test_strategy(self, smas):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ============
        smas: tuple (SMA_S, SMA_M, SMA_L)
            Simple Moving Averages to be considered for the strategy.

        '''

        self.SMA_S = smas[0]
        self.SMA_M = smas[1]
        self.SMA_L = smas[2]

        self.prepare_data(smas=smas)
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        self.print_performance()

    def prepare_data(self, smas):
        ''' Prepares the Data for Backtesting.
        '''
        ########################## Strategy-Specific #############################

        data = self.data[["Close", "returns"]].copy()
        data["SMA_S"] = data.Close.rolling(window=smas[0]).mean()
        data["SMA_M"] = data.Close.rolling(window=smas[1]).mean()
        data["SMA_L"] = data.Close.rolling(window=smas[2]).mean()

        data.dropna(inplace=True)

        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)

        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        ##########################################################################

        self.results = data

    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''

        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc

        self.results = data

    def plot_results(self):
        '''  Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    def optimize_strategy(self, SMA_S_range, SMA_M_range, SMA_L_range, metric="Multiple"):
        '''
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).

        Parameters
        ============
        SMA_S_range: tuple
            tuples of the form (start, end, step size).

        SMA_M_range: tuple
            tuples of the form (start, end, step size).

        SMA_L_range: tuple
            tuples of the form (start, end, step size).

        metric: str
            performance metric to be optimized (can be "Multiple" or "Sharpe")
        '''

        self.metric = metric

        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe

        SMA_S_range = range(*SMA_S_range)
        SMA_M_range = range(*SMA_M_range)
        SMA_L_range = range(*SMA_L_range)

        combinations = list(product(SMA_S_range, SMA_M_range, SMA_L_range))

        performance = []
        for comb in combinations:
            self.prepare_data(smas=comb)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))

        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["SMA_S", "SMA_M", "SMA_L"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum).
        '''

        best = self.results_overview.nlargest(1, "performance")
        SMA_S = best.SMA_S.iloc[0]
        SMA_M = best.SMA_M.iloc[0]
        SMA_L = best.SMA_L.iloc[0]
        perf = best.performance.iloc[0]
        print("SMA_S: {} | SMA_M: {} | SMA_L : {} | {}: {}".format(SMA_S, SMA_M, SMA_L, self.metric, round(perf, 5)))
        self.test_strategy(smas=(SMA_S, SMA_M, SMA_L))

    def add_sessions(self, vizualize=False):

        if self.results is None:
            print('Run test_strategy first')

        data = self.results.copy()
        data['session'] = np.sign(data.trades).cumsum().shift().fillna(0)
        data['session_compound'] = data.groupby('session').strategy.cumsum().apply(np.exp) - 1
        self.results = data

    def add_leverage(self, leverage, report=True):

        self.add_sessions()
        self.leverage = leverage

        data = self.results.copy()
        data['simple_ret'] = np.exp(data.strategy)
        data['eff_lev'] = leverage * (1 + data.session_compound / (1 + data.session_compound * leverage))
        data.eff_lev.fillna(leverage, inplace=True)
        data.loc[data.trades != 0, 'eff_lev'] = leverage
        levered_returns = data.eff_lev.shift() * data.simple_ret
        levered_returns = np.where(levered_returns < -1, -1, levered_returns)
        data['strategy_levered'] = levered_returns
        data['cstrategy_levered'] = data.strategy_levered.add(1).cumprod()

        self.results = data

        if report:
            self.print_performance(leverage=True)

    ############################## Performance ######################################

    def print_performance(self, leverage=False):
        ''' Calculates and prints various Performance Metrics.
        '''

        data = self.results.copy()

        if leverage:
            to_analyze = np.log(data.strategy_levered.add(1))
        else:
            to_analyze = data.strategy

        strategy_multiple = round(self.calculate_multiple(data.strategy), 6)
        bh_multiple = round(self.calculate_multiple(data.returns), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(data.strategy), 6)
        ann_mean = round(self.calculate_annualized_mean(data.strategy), 6)
        ann_std = round(self.calculate_annualized_std(data.strategy), 6)
        sharpe = round(self.calculate_sharpe(data.strategy), 6)

        print(100 * "=")
        print("TRIPLE SMA STRATEGY | INSTRUMENT = {} | SMAs = {}".format(self.symbol,
                                                                         [self.SMA_S, self.SMA_M, self.SMA_L]))
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))

        print(100 * "=")

    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1

    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)


class Long_Short_Backtester:
    ''' Class for the vectorized backtesting of simple Long-Short trading strategies.

    Attributes
    ============
    filepath: str
        local filepath of the dataset (csv-file)
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade


    Methods
    =======
    get_data:
        imports the data.

    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).

    prepare_data:
        prepares the data for backtesting.

    run_backtest:
        runs the strategy backtest.

    plot_results:
        plots the cumulative performance of the trading strategy compared to buy-and-hold.

    optimize_strategy:
        backtests strategy for different parameter values incl. optimization and reporting (wrapper).

    find_best_strategy:
        finds the optimal strategy (global maximum).


    print_performance:
        calculates and prints various performance metrics.

    '''

    def __init__(self, filepath, symbol, start, end, tc):

        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def __repr__(self):
        return "Long_Short_Backtester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)

    def get_data(self):
        ''' Imports the data.
        '''
        raw = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        raw = raw.loc[self.start:self.end].copy()
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        self.data = raw

    def test_strategy(self, smas):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ============
        smas: tuple (SMA_S, SMA_M, SMA_L)
            Simple Moving Averages to be considered for the strategy.

        '''

        self.SMA_S = smas[0]
        self.SMA_M = smas[1]
        self.SMA_L = smas[2]

        self.prepare_data(smas=smas)
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        self.print_performance()

    def prepare_data(self, smas):
        ''' Prepares the Data for Backtesting.
        '''
        ########################## Strategy-Specific #############################

        data = self.data[["Close", "returns"]].copy()
        data["SMA_S"] = data.Close.rolling(window=smas[0]).mean()
        data["SMA_M"] = data.Close.rolling(window=smas[1]).mean()
        data["SMA_L"] = data.Close.rolling(window=smas[2]).mean()

        data.dropna(inplace=True)

        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)

        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        ##########################################################################

        self.results = data

    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''

        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc

        self.results = data

    def plot_results(self):
        '''  Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    def optimize_strategy(self, SMA_S_range, SMA_M_range, SMA_L_range, metric="Multiple"):
        '''
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).

        Parameters
        ============
        SMA_S_range: tuple
            tuples of the form (start, end, step size).

        SMA_M_range: tuple
            tuples of the form (start, end, step size).

        SMA_L_range: tuple
            tuples of the form (start, end, step size).

        metric: str
            performance metric to be optimized (can be "Multiple" or "Sharpe")
        '''

        self.metric = metric

        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe

        SMA_S_range = range(*SMA_S_range)
        SMA_M_range = range(*SMA_M_range)
        SMA_L_range = range(*SMA_L_range)

        combinations = list(product(SMA_S_range, SMA_M_range, SMA_L_range))

        performance = []
        for comb in combinations:
            self.prepare_data(smas=comb)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))

        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["SMA_S", "SMA_M", "SMA_L"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum).
        '''

        best = self.results_overview.nlargest(1, "performance")
        SMA_S = best.SMA_S.iloc[0]
        SMA_M = best.SMA_M.iloc[0]
        SMA_L = best.SMA_L.iloc[0]
        perf = best.performance.iloc[0]
        print("SMA_S: {} | SMA_M: {} | SMA_L : {} | {}: {}".format(SMA_S, SMA_M, SMA_L, self.metric, round(perf, 5)))
        self.test_strategy(smas=(SMA_S, SMA_M, SMA_L))

    ############################## Performance ######################################

    def print_performance(self):
        ''' Calculates and prints various Performance Metrics.
        '''

        data = self.results.copy()
        strategy_multiple = round(self.calculate_multiple(data.strategy), 6)
        bh_multiple = round(self.calculate_multiple(data.returns), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(data.strategy), 6)
        ann_mean = round(self.calculate_annualized_mean(data.strategy), 6)
        ann_std = round(self.calculate_annualized_std(data.strategy), 6)
        sharpe = round(self.calculate_sharpe(data.strategy), 6)

        print(100 * "=")
        print("TRIPLE SMA STRATEGY | INSTRUMENT = {} | SMAs = {}".format(self.symbol,
                                                                         [self.SMA_S, self.SMA_M, self.SMA_L]))
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))

        print(100 * "=")

    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1

    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)


filepath = "bitcoin.csv"
symbol = "BTCUSDT"
start = "2017-08-17"
end = "2021-10-07"
tc = -0.00085
sma_s = 15
sma_m = 50
sma_l = 200

tester = Long_Short_Backtester(filepath=filepath, symbol=symbol,
                               start=start, end=end, tc=tc)

print(tester.data)
