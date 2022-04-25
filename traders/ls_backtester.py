import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use("seaborn")


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


class Long_Short_Backtester_SMA:
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

    def test_strategy(self, percentiles=None, thresh=None):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ============
        smas: tuple (SMA_S, SMA_M, SMA_L)
            Simple Moving Averages to be considered for the strategy.

        '''

        self.prepare_data(percentiles=percentiles, thresh=thresh)
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        self.print_performance()

    def prepare_data(self, percentiles, thresh):
        ''' Prepares the Data for Backtesting.
        '''
        ########################## Strategy-Specific #############################

        data = self.data[["Close", "Volume", "returns"]].copy()
        data['vol_ch'] = np.log(data.Volume.div(data.Volume.shift(1)))
        data.loc[data.vol_ch > 3, 'vol_ch'] = np.nan
        data.loc[data.vol_ch < -3, 'vol_ch'] = np.nan

        if percentiles:
            self.return_thresh = np.percentile(data.returns.dropna(), [percentiles[0], percentiles[1]])
            self.volume_thresh = np.percentile(data.vol_ch.dropna(), [percentiles[2], percentiles[3]])
        elif thresh:
            self.return_thresh = [thresh[0], thresh[1]]
            self.volume_thresh = [thresh[2], thresh[3]]

        cond1 = data.returns >= self.return_thresh[0]
        cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
        cond3 = data.returns <= self.return_thresh[1]

        data['position'] = 0
        data.loc[cond1 & cond2, 'position'] = 1
        data.loc[cond3 & cond2, 'position'] = -1

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

    def optimize_strategy(self, return_low_range, return_high_range, vol_low_range, vol_high_range, metric="Multiple"):
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

        return_low_range = range(*return_low_range)
        return_high_range = range(*return_high_range)
        vol_low_range = range(*vol_low_range)
        vol_high_range = range(*vol_high_range)

        combinations = list(product(return_low_range, return_high_range, vol_low_range, vol_high_range))

        performance = []
        for comb in combinations:
            self.prepare_data(percentiles=comb, thresh=None)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))

        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["return_low", "return_high",
                                                                                   "volume_low", "volume_high"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum).
        '''

        best = self.results_overview.nlargest(1, "performance")
        return_perc = [best.return_low.iloc[0], best.return_high.iloc[0]]
        volume_perc = [best.volume_low.iloc[0], best.volume_high.iloc[0]]
        perf = best.performance.iloc[0]
        print(
            "Return_Perc: {} | Volume_perc: {} | {}: {}".format(return_perc, volume_perc, self.metric, round(perf, 5)))
        self.test_strategy(percentiles=(return_perc[0], return_perc[1], volume_perc[0], volume_perc[1]))

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
        print("PRICE & VOLUME STRATEGY | INSTRUMENT = {} | THRESHOLDS = {}, {}".format(self.symbol,
                                                                                       self.return_thresh,
                                                                                       self.volume_thresh))
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


class Long_Only_Backtesteer:
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

    def test_strategy(self, percentiles=None, thresh=None):
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ============
        smas: tuple (SMA_S, SMA_M, SMA_L)
            Simple Moving Averages to be considered for the strategy.

        '''

        self.prepare_data(percentiles=percentiles, thresh=thresh)
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        self.print_performance()

    def prepare_data(self, percentiles, thresh):
        ''' Prepares the Data for Backtesting.
        '''

        data = self.data[["Close", "Volume", "returns"]].copy()
        data['vol_ch'] = np.log(data.Volume.div(data.Volume.shift(1)))
        data.loc[data.vol_ch > 3, 'vol_ch'] = np.nan
        data.loc[data.vol_ch < -3, 'vol_ch'] = np.nan

        if percentiles:
            self.return_thresh = np.percentile(data.returns.dropna(), percentiles[0])
            self.volume_thresh = np.percentile(data.vol_ch.dropna(), [percentiles[1], percentiles[2]])
        elif thresh:
            self.return_thresh = thresh[0]
            self.volume_thresh = [thresh[1], thresh[2]]

        cond1 = data.returns >= self.return_thresh
        cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])

        data['position'] = 1
        data.loc[cond1 & cond2, 'position'] = 0

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

    def optimize_strategy(self, return_range, vol_low_range, vol_high_range, metric="Multiple"):
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

        return_range = range(*return_range)
        vol_low_range = range(*vol_low_range)
        vol_high_range = range(*vol_high_range)

        combinations = list(product(return_range, vol_low_range, vol_high_range))

        performance = []
        for comb in combinations:
            self.prepare_data(percentiles=comb, thresh=None)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))

        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["returns", "volume_low",
                                                                                   "volume_high"])
        self.results_overview["performance"] = performance
        self.find_best_strategy()

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum).
        '''

        best = self.results_overview.nlargest(1, "performance")
        return_perc = best.returns.iloc[0]
        volume_perc = [best.volume_low.iloc[0], best.volume_high.iloc[0]]
        perf = best.performance.iloc[0]
        print(
            "Return_Perc: {} | Volume_perc: {} | {}: {}".format(return_perc, volume_perc, self.metric, round(perf, 5)))
        self.test_strategy(percentiles=(return_perc, volume_perc[0], volume_perc[1]))

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
        print("PRICE & VOLUME STRATEGY | INSTRUMENT = {} | THRESHOLDS = {}, {}".format(self.symbol,
                                                                                       self.return_thresh,
                                                                                       self.volume_thresh))
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

percentiles = (90, 5, 20)

tester = Long_Only_Backtesteer(filepath, symbol, start, end, tc)
tester.test_strategy(percentiles)

print(tester.results)

tester.optimize_strategy(return_range=(85, 98, 1),
                         vol_low_range=(2, 16, 1),
                         vol_high_range=(16, 35, 1))

print(tester.results_overview)
print(tester.find_best_strategy())
