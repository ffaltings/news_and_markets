"""
Backtesting the different portfolios with the fitted model
- Types of positions: long (L), short (S), long-short (L-S)
- Allocation types: equally-weighted (EW), value-weighted (VW)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.models.utils.large_pickle_io import read_large_pickle, write_large_pickle
from src.models.utils.finance_preprocess import add_news_to_word_count
from src.models.sestm_model import SESTM_model
from src.models.model_param_search import parameter_selection
import gc
import time
import dateutil.relativedelta
import pickle
import pandas_market_calendars as mcal
import seaborn as sns
from src.models.utils.backtesting_utils import Logger, pnl_metrics, get_business_day, \
                                                         get_trading_weights_random, \
                                                         get_trading_weights_alexandria_bert, \
                                                         random_strategy_pnls



class BackTester:

    def __init__(self, model, start_date="2016-09-01", end_date="2019-11-13",
                 train_periods=10, validation_periods=5, trading_period=1,
                 refitting_frequency=1, recent_news_only=False):
        """

        :param: model: model object, see model.py
        :param start_date: start date of backtesting, format YYYY-MM-DD
        :param end_date: end date of backtesting, format YYYY-MM-DD
        :param train_periods: how many periods are used for the fitting of the model
        :param validation_periods: how many periods are used for the validation of the model
        :param trading_period: how often to get new weights from sentiment model, in days
        (in benchmark paper they recalculate weights each day, that's why default value is set to 1,
        i.e. we recalculate weights daily)
        :param refitting_frequency: refit the model every n months. Only used when running backtest on SESTM model.
        :param recent_news_only: if true, for trading on day t consider news published between yesterday 4PM and today
            9AM. If false, for trading on day t consider news published between day t-1 9.30AM and today 9AM.
            Difference between yesterday and day t-1 is important for weekends and holidays.

        """
        self.model = model
        self.start_date = start_date
        self.end_date = end_date
        self.train_periods = train_periods
        self.validation_periods = validation_periods
        self.trading_period = trading_period
        self.refitting_frequency = refitting_frequency
        self.recent_news_only = recent_news_only

        # set trading dates, exclude holidays
        self.trading_dates = [x.date() for x in mcal.get_calendar('NYSE').valid_days(start_date=self.start_date,
                                                                                     end_date=self.end_date)]

        # # set refitting dates to first business days of the month, again excluding holidays
        self.refitting_dates = [get_business_day(d).date()
                                for d in pd.date_range(self.start_date, self.end_date, freq='BMS')]

        # start trading in Jan 2018, hence +1
        # TODO: start of trading is now adjusted here via + at the end. Refactor and move this to the wrapper
        self.start_of_trading = self.refitting_dates[self.train_periods + self.validation_periods + 1]

        # take refitting frequency (monthly) into account
        self.refitting_dates = self.refitting_dates[self.refitting_dates.index(self.start_of_trading)::self.refitting_frequency]

        # refitting dates before code refactoring on 15.4.2020 (better trading performance with it)
        # self.refitting_dates = pd.date_range(self.start_date, self.end_date, freq='B')
        # self.start_of_trading = self.refitting_dates[(self.train_periods + self.validation_periods) * 22]
        # self.refitting_dates = self.refitting_dates[(self.train_periods + self.validation_periods)*22::22]

    @staticmethod
    def get_news_labels(today_date, df_news):
        """
        Get subset of labeled news prior to today_date.
        Used when running backtest without CV.

        :param today_date:
        :param df_news:
        :return: labels, indices of relevant news
        """
        df_news = df_news[(df_news["date_NYC"] < today_date) & (df_news["LABEL"].notnull())]
        labels = df_news.LABEL
        ind = list(df_news.index)
        return labels, ind

    def get_news_labels_CV(self, today_date, word_count, index_df, logger, expanding):
        """
        Get train and validation data-sets for grid-search CV.

        :param today_date:
        :param word_count:
        :param index_df:
        :param logger:
        :param expanding:
        :return:
        """
        if expanding:
            start_training_period = self.start_date
        else:
            start_training_period = today_date - dateutil.relativedelta.relativedelta(months=self.warmup_periods)
        logger.append("Start of training date: {}".format(start_training_period))
        end_training_period = today_date - dateutil.relativedelta.relativedelta(months=self.validation_periods)
        logger.append("Start of validation date: {}".format(end_training_period))

        if expanding:
            train_idx = [i for i, date in enumerate(index_df['date_NYC']) if date < end_training_period]
        else:
            train_idx = [i for i, date in enumerate(index_df['date_NYC']) if
                                    (date >= start_training_period and date < end_training_period)]
        valid_idx = [i for i, date in enumerate(index_df['date_NYC']) if
                                    (date >= end_training_period and date < today_date)]

        logger.append('Number of training samples: {}'.format(len(train_idx)))
        logger.append('Number of validation samples: {}'.format(len(valid_idx)))

        X_train = word_count[train_idx, :]
        y_train = index_df.loc[train_idx, 'LABEL'].values
        X_valid = word_count[valid_idx, :]
        y_valid = index_df.loc[valid_idx, 'LABEL'].values

        return X_train, y_train, X_valid, y_valid

    def run_backtest(self, open_prices_df, news_matrix, index_matrix,
                     weights_method, strategy, weights_pickle_path=None,
                     index_matrix_with_sentiment_path=None, nr_companies=50,
                     logger=None, CV=False, words=None, optimizer="scipy",
                     dt=None, expandingCV=False, alexandria=None, random_weights=False, bert=None):
        """
        :param open_prices: dataframe with open-to-open prices for sp500 companies
        :param news_matrix: word count matrix
        :param index_matrix: index matrix with financial labels
        :param weights_method: equal-weighted (EW), value-weighted (VW), equal-weighted all companies (EWall)
                or company ticker
        :param strategy: long-short (LS), long (L) or short (S)
        :param weights_pickle_path: path for saving weights csv
        :param index_matrix_with_sentiment_path: path for saving index matrix (with new sentiment column)
        :param nr_companies: number of companies we invest in each day (for LS portfolio,
                for L and S we have nr_companies/2)
        :param logger: logging object
        :param CV: whether or not to run CV on each refitting date
        :param words: columns of word_count matrix, for the printing of sentiment words in fitted models
        :param optimizer: which optimizer to use in model's predict function, scipy or int (size of the grid in grid search)
        :param dt: datetime of start of backtesting (for filenames)
        :param expandingCV: whether or not to do expanding version of time series CV
        :param alexandria: whether or not to use Alexandria sentiment data. None or path to Alexandria index dataframe
        :param random_weights: get random trading weights

        """

        # some asserts and logging stuff
        assert strategy in ["LS", "S", "L", None], "Unknown strategy!"
        assert nr_companies % 2 == 0
        if not random_weights:
            logger.append('Running pipeline for {}-{}-{}{}'.format(strategy, weights_method, "CV" if CV else "",
                                                                   '-expanding' if expandingCV else ""))
            logger.append("Trading period (multiple day portfolios): {}".format(self.trading_period))
            logger.append('Nr. of companies param: {}'.format(nr_companies))
        if weights_method in ["EW", "VW"] and not CV and not alexandria and not random_weights and not bert:
            logger.append("Refitting the model every {} month{}!".format(self.refitting_frequency,
                                                               's' if self.refitting_frequency > 1 else ''))
            logger.append("Alpha: {}".format(self.model.alphaPlus))
            logger.append("Kappa: {}".format(self.model.kappa))
            logger.append("Lambda: {}". format(self.model.penaltyLambda))
            logger.append("Lambda regularization: {}". format(self.model.lambda_reg))

        # ================= 1) filter open prices df, init weights df  =================

        # filter open price df and calculate returns
        open_prices = open_prices_df.loc[datetime.strptime(self.start_date, '%Y-%m-%d').date():
                                      datetime.strptime(self.end_date, '%Y-%m-%d').date()]
        assert len(open_prices.index) == len(self.trading_dates), "Holidays..."
        open_prices = open_prices.loc[self.start_of_trading:]
        open_returns = open_prices.pct_change().fillna(0)
        assert weights_method in ["EW", "VW", "EWall"] + list(open_prices.columns), "Unknown weights method!"

        # define weights matrix which will get populated as we trade
        # to get trading results in the end we will multiply (elem-wise) lagged weights df and open returns df
        weights = pd.DataFrame(data=np.nan, columns=open_prices.columns, index=open_prices.index)

        # ================= 2) main trading and fitting loop  =================

        # here we separate between dtindex and trading days. Distinction is important for multiple day portfolios,
        # since there we trade only every n days!
        start_of_trading_index = self.trading_dates.index(self.start_of_trading)
        dtindex = self.trading_dates.copy()
        self.trading_dates = self.trading_dates[start_of_trading_index::self.trading_period]

        # iterate over days and trade/refit
        for i in range(len(dtindex[start_of_trading_index:])):
            today = dtindex[start_of_trading_index + i]
            yesterday = dtindex[start_of_trading_index + i - 1]
            if self.recent_news_only:
                # last = today + timedelta(days=-1)
                last = dtindex[start_of_trading_index + i - self.trading_period]
            else:
                last = dtindex[start_of_trading_index + i - self.trading_period]

            if today in self.refitting_dates and weights_method in ["EW", "VW"] and not alexandria \
                    and not random_weights and not bert:

                # fit the model
                logger.append('{}: Refitting the model'.format(today))
                start_time = time.time()
                if CV:  # run CV
                    X_train, y_train, X_valid, y_valid = self.get_news_labels_CV(today, news_matrix, index_matrix,
                                                                                 logger, expandingCV)
                    self.model, _ = parameter_selection(X_train, y_train, X_valid, y_valid, optimizer, logger,
                                                        to_rank=self.model.to_rank, verbose=True, refit=True)
                    logger.append("Chosen alpha: {}".format(self.model.alphaMinus))
                    logger.append("Chosen kappa: {}".format(self.model.kappa))
                    logger.append("Chosen lambda: {}".format(self.model.penaltyLamda))
                    logger.append("Chosen lambda_reg: {}".format(self.model.lambda_reg))
                else:  # fit model on all past data prior to today, w/o validation
                    labels_, ind_ = self.get_news_labels(today, index_matrix)
                    self.model.fit(news_matrix[ind_, :], labels_)
                end_time = time.time()
                logger.append("Fitting the model took {} seconds!".format(round(end_time-start_time, 2)))

                # logging stuff, later parsed and used to get wordclouds etc.
                positive_words = [words[i] for i in list(self.model.positive_set)]
                negative_words = [words[i] for i in list(self.model.negative_set)]
                logger.append("Positive words: {}".format(str(positive_words)))
                logger.append("Negative words: {}".format(str(negative_words)))

                words_model = np.array([words[i] for i in self.model.S_hat])
                tone = np.array(self.model.O_hat[:, 0] - self.model.O_hat[:, 1])[:, 0] / 2
                logger.append("Tone vector: {}".format(str(list(zip(words_model, tone)))))

            if today in self.trading_dates:  # get new weights
                if not random_weights:
                    logger.append('{}: Recalculating weights'.format(today))

                # sentiment weights
                if weights_method == "EW" or weights_method == "VW":
                    if alexandria or bert:
                        weights_today = get_trading_weights_alexandria_bert(today, last, weights_method, strategy,
                                                                            nr_companies, weights.columns,
                                                                            index_matrix, logger, self.recent_news_only)

                        weights.loc[today, :] = weights_today
                    elif random_weights:
                        weights_today = get_trading_weights_random(strategy, nr_companies, weights.columns,
                                                                   logger)
                        weights.loc[today, :] = weights_today
                    else:
                        weights_today, index_today, pred_sentiment_today = self.model.get_trading_weights(today,
                                                                        last, weights_method, strategy,
                                                                        nr_companies, weights.columns, index_matrix,
                                                                        news_matrix, logger,
                                                                        self.recent_news_only)
                        weights.loc[today, :] = weights_today

                        # storing predicted sentiment per news to the index matrix
                        if (index_today is not None) and (pred_sentiment_today is not None):
                            index_matrix.loc[index_today, "pred_sentiment"] = pred_sentiment_today
                            index_matrix.loc[index_today, 'trading_day_index'] = i

                # equal weights all companies
                elif weights_method == "EWall":
                    weights.loc[today, :] = (1/weights.shape[1]) * np.ones(weights.shape[1])

                # invest in a single company, serves as testing for correctness of backtesting pipeline
                else:
                    weights.loc[today, :] = [0] * weights.shape[1]
                    weights.loc[today, weights_method] = 1

            else:  # only gets executed in case self.trading_period > 1

                if weights_method != "VW":
                    if not random_weights:
                        logger.append('{}: Keeping same weights (using weights from {}).'.format(today, yesterday))
                    weights.loc[today, :] = weights.loc[yesterday, :]
                else:  # re-balance the weights (in case of value-weighted portfolio)
                    if not random_weights:
                        logger.append('{}: Rebalance to preserve weights w.r.t. company size in VW portfolio.'.format(today))
                    weights.loc[today, :] = weights.loc[last, :] * (1 + open_returns.loc[today, :]) \
                                            / (1 + (weights.loc[last, :] * open_returns.loc[today, :]).sum())

        # ================= 3) save weights and index matrix with sentiment =================

        # save weights per day, later used to generate various plots (Figure 7, Figure 8) and calculate trading metrics
        if weights_pickle_path:
            weights_filename = weights_pickle_path + weights_method + "_" + strategy + "_" + \
                                        dt.strftime("%Y-%m-%d_%H-%M-%S") + "_weights.p"
            weights_dict = {"weights_df": weights, "open_prices_index": dtindex[start_of_trading_index:]}
            pickle.dump(weights_dict, open(weights_filename, "wb"))

        # save index matrix with new sentiment column
        if index_matrix_with_sentiment_path and not alexandria and not random_weights and not bert:
            index_matrix_filename = index_matrix_with_sentiment_path + weights_method + "_" + strategy + "_" + \
                               dt.strftime("%Y-%m-%d_%H-%M-%S") + "_index_matrix.p"
            write_large_pickle(index_matrix, index_matrix_filename)

        # ================= 4) trading plots and metrics =================

        # pnl calculation (day 1)
        pnl = (weights.shift(1) * open_returns).sum(axis=1)

        # pnl plot
        if not random_weights:
            pnl.cumsum().plot(title="pnl of {}{} portfolio (day 1)".format(strategy + "-"
                              if strategy else "", weights_method), rot=90)
            plt.show()

        # trading metrics
        sharpe, _, _, mdd, ann_ret, ret, _, _ = pnl_metrics(pnl=pnl, print_=False)
        logger.append("Annualized Sharpe: {}".format(sharpe))
        logger.append("Annualized return: {}".format(ann_ret))
        logger.append("MDD: {}".format(mdd))
        logger.append("Return: {}".format(ret))

        return [sharpe, ann_ret, mdd, ret], pnl


def run_backtest_wrapper(open_prices_path, word_count_matrix_path, index_matrix_path,
                         new_indices_path, logger_path, weights_pickle_path,
                         index_matrix_with_sentiment_path, weights_method,
                         strategy, validation_periods, CV, nr_companies, model_alpha_init,
                         model_kappa_init, model_lambda_init, optimizer, expandingCV, alexandria,
                         trading_periods, start_date, end_date, fin_labels_type,
                         to_rank, model_lambda_reg_init, refitting_frequency,
                         recent_news_only, bert, train_periods=10):
    """
    Wrapper function for running the backtest. The following steps get executed (in case of SESTM model):
    - init of model object
    - init of logger object
    - read in sp500 open prices df
    - loading of the data (word_count matrix, index matrix)
    - (in case of align_v1: new rows are added to the word_count matrix)
    - in case of 2019 data:
        preprocessing of datetime in index matrix (new column is added, which is a merge of date and time columns)
    - init of backtest object
    - running the backtest

    :param open_prices_path: path to dataframe with open-to-open prices for sp500 companies
    :param word_count_matrix_path: path to word count matrix pickle
    :param index_matrix_path: path to index matrix pickle
    :param new_indices_path: path to new indices pickle (contains news that need to be added to word count
                        matrix due to multiple alignment). Only relevant if alignment used is align_v1.
    :param logger_path: path to where backtesting log file should be saved
    :param weights_pickle_path: path to where weights dataframe should be saved (pickle format)
    :param index_matrix_with_sentiment_path: path for saving index matrix (with new sentiment column)
    :param weights_method: EW or VW
    :param strategy: LS, L, S
    :param validation_periods: how many periods are used for the validation of the model
    :param CV: whether or not to run CV on each refitting date
    :param nr_companies: number of companies we invest in each day for each leg of the portfolio, i.e.
            for LS portfolio we have 2*nr_companies
    :param model_alpha_init: initial alpha parameter of the model (if not running CV, this value is used for all
                        fitting periods)
    :param model_kappa_init: initial kappa parameter of the model (if not running CV, this value is used for all
                        fitting periods)
    :param model_lambda_init: initial lambda parameter of the model (if not running CV, this value is used for all
                        fitting periods)
    :param optimizer: which optimizer to use in model's predict function, scipy or int (size of the grid in grid search)
    :param expandingCV: whether or not to do expanding version of time series CV
    :param alexandria: whether or not to use Alexandria sentiment data
    :param trading_periods: nr. of days between that we hold our positions, for daily trading set to 1
    :param start_date: starting date of trading
    :param end_date: end date of trading
    :param fin_labels_type: normal, beta_close, beta_adj_close (type of financial labels to use)
    :param to_rank: if true model outputs ranked sentiment. Else it outputs raw sentiment.
    :param model_lambda_reg_init: regularization hyperparam when fitting O matrix
    :param refitting_frequency: refit model every n months. Used only for SESTM runs.
    :param recent_news_only: if true, for trading on day t consider news published between yesterday 4PM and today
            9AM. If false, for trading on day t consider news published between day t-1 9.30AM and today 9AM.
            Difference between yesterday and day t-1 is important for weekends and holidays.
    :param train_periods: how many periods are used for the fitting of the model

    """

    dt = datetime.now()

    # read in open prices
    open_prices = pd.read_csv(open_prices_path, delimiter=',', parse_dates=["date"])
    open_prices['date'] = open_prices['date'].apply(lambda x: x.date())
    open_prices.set_index("date", drop=True, inplace=True)
    open_prices.drop(["Unnamed: 0"], axis=1, inplace=True)  # drop old index column
    open_prices.dropna(axis=0, thresh=500, inplace=True)

    # SESTM model or Alexandria
    if weights_method in ["EW", "VW"]:

        assert (alexandria is None) or (bert is None), "Can not run alexandria and bert backtest at once!"

        # set up the logger
        logger = Logger(logger_path, weights_method, strategy, dt)

        # Load the news data and drop rows without labels
        if alexandria:
            # alexandria
            index_df = pd.read_csv(alexandria, index_col=0, parse_dates=["date_NYC", "datetime_NYC"])
            index_df["time_NYC"] = index_df["time_NYC"].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())
            logger.append("Read in Alexandria dataset. Number of news {}".format(len(index_df)))

            word_count = None
            words = None
            model_ = None

        elif bert:

            index_df = pd.read_csv(bert[0], index_col=0, parse_dates=["date_NYC", "datetime_NYC"])
            index_df["time_NYC"] = index_df["time_NYC"].apply(lambda x: datetime.strptime(x[:8], '%H:%M:%S').time())
            index_df['sentiment'] = index_df[bert[1]]
            index_df = index_df[['TICKER', "date_NYC", "datetime_NYC", 'time_NYC', 'sentiment']]
            n = len(index_df)
            index_df.dropna(inplace=True)
            logger.append("Dropped {} rows due to nan signal.".format(n-len(index_df)))
            logger.append("Read in BERT dataset, signal: {}. Number of news {}".format(bert[1], len(index_df)))

            word_count = None
            words = None
            model_ = None

        else:  # SESTM model

            # initialize the model
            model_ = SESTM_model(alphaPlus=model_alpha_init, alphaMinus=model_alpha_init, kappa=model_kappa_init,
                                 penaltyLambda=model_lambda_init, lambda_reg=model_lambda_reg_init,
                                 to_rank=to_rank, optimizer=optimizer)

            logger.append("Reading in the data: {} !".format(word_count_matrix_path))
            logger.append("Alignment used: {}".format(index_matrix_path))
            logger.append("Optimizer used: {}".format(optimizer))
            logger.append('SESTM model output: {}'.format('raw sentiment' if not to_rank else 'ranked sentiment'))
            data = read_large_pickle(word_count_matrix_path)
            ind_mat = read_large_pickle(index_matrix_path)

            logger.append('Labels used: {}'.format(fin_labels_type))
            if fin_labels_type == 'beta_close':
                ind_mat['LABEL'] = ind_mat['beta_labels_close']
            elif fin_labels_type == 'beta_adj_close':
                ind_mat['LABEL'] = ind_mat['beta_labels_adjusted_close']

            if new_indices_path:
                assert "align_v1" in index_matrix_path, "New indices only relevant for alignment align_v1!"
                new_indices = pickle.load(open(new_indices_path, "rb"))
                data["count_matrix"] = add_news_to_word_count(data["count_matrix"], new_indices)

            logger.append('Number of observations: ', data['count_matrix'].shape[0])
            logger.append('Number of words: ', data['count_matrix'].shape[1])

            idx_to_drop = ~ind_mat['LABEL'].isna().values
            logger.append('Number of rows with NaN Label: ', sum(~idx_to_drop))

            if domains_to_exclude:
                idx_to_drop_domains = ind_mat['url'].isin(domains_to_exclude).values
                logger.append('Number of rows from shady domains: {}. Domains excluded: {}'.format(
                              sum(idx_to_drop_domains), str(domains_to_exclude)))
                idx_to_drop = ~((ind_mat.LABEL.isna()) | (ind_mat.url.isin(domains_to_exclude))).values
            
            word_count = data['count_matrix'][idx_to_drop, :]
            index_df = ind_mat[idx_to_drop].reset_index(drop=True)
            assert index_df.shape[0] == word_count.shape[0]
            logger.append('Number of news available for trading: {}'.format(index_df.shape[0]))

            words = data['columns']
            del data
            gc.collect()

            if not ('2020' in index_matrix_path):  # precompute date and time combine in case of index matrix from 2019
                def to_datetime(x):
                    if x.date_NYC:
                        return datetime.combine(x.date_NYC, x.time_NYC)
                    else:
                        return None

                start_time = time.time()
                index_df["datetime_NYC"] = index_df.apply(to_datetime, axis=1)
                end_time = time.time()
                logger.append("Precomputing datetime took {} seconds!".format(round(end_time - start_time, 2)))
                print(index_df.head())

            # for storing predicted sentiment per news to the index matrix
            if index_matrix_with_sentiment_path is not None:
                index_df["pred_sentiment"] = np.nan
                index_df['trading_day_index'] = np.nan

            # some more asserts
            assert word_count.shape[1] == len(words)

        # init backtest and run
        backtest = BackTester(model=model_, train_periods=train_periods, validation_periods=validation_periods,
                              trading_period=trading_periods, start_date=start_date, end_date=end_date,
                              refitting_frequency=refitting_frequency, recent_news_only=recent_news_only)
        return backtest.run_backtest(open_prices_df=open_prices, news_matrix=word_count, index_matrix=index_df,
                              weights_method=weights_method, strategy=strategy, logger=logger,
                              CV=CV, nr_companies=nr_companies, weights_pickle_path=weights_pickle_path,
                              index_matrix_with_sentiment_path=index_matrix_with_sentiment_path, words=words,
                              optimizer=optimizer, dt=dt, expandingCV=expandingCV, alexandria=alexandria, bert=bert)

    # EWall or single company portfolio
    else:
        # set up the logger
        logger = Logger(logger_path, weights_method, "", dt)

        # init backtest and run
        backtest = BackTester(model=None, train_periods=train_periods, validation_periods=validation_periods,
                              trading_period=trading_periods, start_date=start_date, end_date=end_date)
        return backtest.run_backtest(open_prices_df=open_prices, news_matrix=None, index_matrix=None,
                              weights_method=weights_method, strategy=None, logger=logger, dt=dt)


def backtest_plot_calculate_metrics(open_prices_path, backtest_weights_df, lag_start=-1, lag_end=1,
                                    spy_path=None, end_date=None,  figure_7=False, figure_8=False, print_metrics=False,
                                    trading_metrics_path=None, daily_returns_path=None, random_pnls_path=None,
                                    fontsize=20, figure_7_title=None):
    """
    Based on the weights dataframe (gets saved during backtesting) produces Figure 7
    and calculates some metrics from Table 3 (see benchmark paper).
    Can also produce Figure 8.

    :param open_prices_path: path to dataframe with open-to-open prices for sp500 companies
    :param backtest_weights_df: path to pickle object with weights dataframe and trading days index
    :param lag_start:
    :param lag_end:
    :param spy_path: path to dataframe with open-to-open prices for sp500 index
    :param end_date: date till which we are plotting results
    :param figure_7: whether or not to plot Figure 7
    :param figure_8: whether or not to plot Figure 8
    :param trading_metrics_path: path to save trading metrics .csv file. If None, metrics are not saved.
    :param daily_returns_path: path to save daily returns .csv file. If None, daily returns are not saved.
    :param random_pnls_path: Path to random pnls pickle file (generated using function random_backtest)
    """

    sns.set()
    plt.figure(figsize=(16, 9))


    # init lags, trading dataframes
    backtest_name = backtest_weights_df[-35:-10]
    lags = list(range(lag_start, lag_end + 1))
    pnls = []
    metrics_df, std_returns = pd.DataFrame(columns=["Sharpe", "Daily Avg Return [bps]",
                                                    "Max. Drawdown", "conf int [bps]", "ann return [%]", "ann vol [%]"]), []
    daily_returns = pd.DataFrame(columns=["Day {}".format(i) for i in lags] + ["SPY"])

    # ==================== 1) read in open prices df, trading weights df, (optional) spy prices df ================

    # read in trading weights
    weights_dict = pickle.load(open(backtest_weights_df, "rb"))
    weights = weights_dict["weights_df"]
    dtindex_trading = weights_dict["open_prices_index"]

    del weights_dict

    # read in open price and calculate returns
    open_prices = pd.read_csv(open_prices_path, delimiter=',', parse_dates=["date"])
    open_prices.set_index("date", drop=True, inplace=True)
    open_prices.drop(["Unnamed: 0"], axis=1, inplace=True)
    open_prices = open_prices.reindex(dtindex_trading)
    open_returns = open_prices.pct_change().fillna(0)

    if end_date:  # only plot till end_date
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        weights = weights.loc[:end_date, :]
        open_returns = open_returns.loc[:end_date, :]

    # read in spy, calculate returns and spy trading metrics
    if spy_path:
        spy_prices = pd.read_csv(spy_path, parse_dates=["Date"])
        spy_prices.set_index("Date", drop=True, inplace=True)
        spy_prices = spy_prices.reindex(dtindex_trading)
        spy_returns = spy_prices.pct_change().fillna(0)
        if end_date:  # only plot till end_date
            spy_returns = spy_returns.loc[:end_date, :]
        if figure_7:
            spy_returns.Open.cumsum().plot(label="SPY")
        daily_returns["SPY"] = spy_returns.Open
        sharpe, avg_return, std_return, mdd, ann_return, _, conf_int, ann_vol = pnl_metrics(pnl=spy_returns.Open, spy_returns=None,
                                                                          print_=print_metrics)
        metrics_df.loc["SPY", :] = {"Sharpe": round(sharpe, 2),
                                    "Daily Avg Return [bps]": round(avg_return, 2),
                                    "Max. Drawdown": round(mdd, 2),
                                    "conf int [bps]": conf_int,
                                    "ann return [%]": ann_return,
                                    "ann vol [%]": ann_vol}

    # ==================== 2) turnover calculation, number of companies held per day ================

    # nr. of companies held per each trading day
    nr_companies_df = weights.shape[1] - weights.eq(0).sum(axis=1)
    if print_metrics:
        print('Nr. of companies we invest in per day:')
        print(nr_companies_df.value_counts())

    # turnover calculation
    if 'EW_LS' in backtest_weights_df:
        # here we work with wealth of 2 (we invest 1 in long leg and 1 in short leg)
        denominator = 4 * len(open_returns)
    else:
        # for L and S portfolios we work with wealth 1
        denominator = 2 * len(open_returns)
    turnover = abs(weights.shift(1) - (1 + open_returns.shift(1)) * weights).sum(axis=1).sum() / denominator
    assert 0 <= turnover <= 1
    if print_metrics:
        print("Turnover: {}".format(round(turnover, 2)))

    # ==================== 3) daily pnls and trading metrics ================

    for lag in lags:
        daily_pnl = (weights.shift(lag) * open_returns).sum(axis=1)
        if figure_7:
            # [update 16.6.] dividing pnls by standard deviations of returns does not seem to make sense for plotting purposes?
            (daily_pnl.cumsum()).plot(label="Day {}".format(lag))
        daily_returns["Day {}".format(lag)] = daily_pnl
        if print_metrics:
            print("--------Metrics for Day {}---------".format(lag))
        sharpe, avg_return, std_return, mdd, ann_return, _, conf_int, ann_vol = pnl_metrics(pnl=daily_pnl, spy_returns=spy_returns,
                                                                print_=print_metrics)
        metrics_df.loc["Day {}".format(lag), :] = {"Sharpe": round(sharpe, 3),
                                                   "Daily Avg Return [bps]": round(avg_return, 3),
                                                   "Max. Drawdown": round(mdd, 3),
                                                   "conf int [bps]": conf_int,
                                                    "ann return [%]": ann_return,
                                                    "ann vol [%]": ann_vol}
        std_returns.append(std_return)

    if trading_metrics_path:
        metrics_df.to_csv(trading_metrics_path + backtest_name + "_trading_metrics.csv")
    if daily_returns_path:
        daily_returns.to_csv(daily_returns_path + backtest_name + '_daily_returns.csv')

    if print_metrics:
        print(metrics_df)

    if random_pnls_path:
        assert figure_7
        df_random = random_strategy_pnls(random_pnls_path, end_date=end_date)

        df_random.mean(axis=1).plot(color='grey', label="Random", linestyle='dashed')
        # plot mu + sigma (and not confidence interval :) )
        plt.fill_between(df_random.index,
                         (df_random.mean(axis=1) + df_random.std(axis=1)).values,
                         (df_random.mean(axis=1) - df_random.std(axis=1)).values,
                         color='black', alpha='0.175')

    if figure_7:
        # plt.xticks(rotation=60)
        # plt.axhline(y=0.0, color="black", linestyle='dashed', alpha=0.2)
        plt.xticks(fontsize=fontsize, rotation=20)
        plt.legend(loc="upper left", fontsize=fontsize)
        plt.ylabel("Cumulative Return", fontsize=fontsize)
        plt.xlabel("Date", fontsize=fontsize)
        if figure_7_title:
            plt.savefig(figure_7_title, format="pdf")
        plt.show()

    # ==================== 4) plot Figure 8 ================

    if figure_8:
        metrics_df.drop(metrics_df.head(1).index, inplace=True)  # drop SPY metrics for Figure 8
        avg_return_daily = metrics_df["Daily Avg Return [bps]"].values
        n = len(daily_returns)
        std_returns = [1.96*x/np.sqrt(n) for x in std_returns]
        right_ = [x + y for x,y in zip(avg_return_daily, std_returns)]
        left_ = [x - y for x, y in zip(avg_return_daily, std_returns)]
        plt.plot(lags, metrics_df["Daily Avg Return [bps]"].values)
        plt.plot(lags, right_, color="blue", linestyle="dashed")
        plt.plot(lags, left_, color="blue", linestyle="dashed")
        plt.fill_between(lags, left_, right_, color='blue', alpha='0.3')
        plt.xticks(lags, labels=["Day {}".format(i) for i in lags], rotation=90)
        # plt.axvline(x=0.82, color="red", linestyle="dashed", label="Absorption point")
        # plt.legend(loc="upper right")
        plt.ylabel("Daily Avg Return [bps]")
        plt.axhline(y=0.0, color="black", linestyle='dashed')
        plt.show()

    return metrics_df, daily_returns, turnover


def random_backtest(open_prices_path, strategy, nr_companies, trading_periods, start_date, end_date,
                    weights_method, logger_path, N=300):
    """
    Baseline for LS trading strategies. Each trading day pick nr_companies for going long and nr_companies for going
    short (never go long and short in the same company). Repeat N times.

    :param open_prices_path:
    :param strategy:
    :param nr_companies:
    :param trading_periods:
    :param start_date:
    :param end_date:
    :param N:
    :return:
    """

    # read in open prices
    open_prices_df = pd.read_csv(open_prices_path, delimiter=',', parse_dates=["date"])
    open_prices_df['date'] = open_prices_df['date'].apply(lambda x: x.date())
    open_prices_df.set_index("date", drop=True, inplace=True)
    open_prices_df.drop(["Unnamed: 0"], axis=1, inplace=True)  # drop old index column
    open_prices_df.dropna(axis=0, thresh=500, inplace=True)

    # set up logger
    dt = datetime.now()
    logger = Logger(logger_path, weights_method, strategy, dt)
    logger.append("RANDOM TRADING WEIGHTS. Not reading in any news.")
    logger.append('Running pipeline for {}-{}'.format(strategy, weights_method))
    logger.append("Trading period (multiple day portfolios): {}".format(trading_periods))
    logger.append('Nr. of companies param: {}'.format(nr_companies))

    # set to None things not used when doing random weights
    index_df = None
    word_count = None
    words = None
    model_ = None

    # unroll backtest N times
    # TODO: currently need to init backtest object every time (to restart trading_dates list). Refactor backtest
    #  code so that will not be necessary.
    metrics = []
    for i in range(N):
        logger.append("BACKTEST {}".format(i + 1))
        backtest = BackTester(model=model_, trading_period=trading_periods, start_date=start_date, end_date=end_date)
        metrics_i, pnl_i = backtest.run_backtest(open_prices_df=open_prices_df, news_matrix=word_count, index_matrix=index_df,
                                          weights_method=weights_method, strategy=strategy, logger=logger,
                                          nr_companies=nr_companies, dt=dt, random_weights=True)
        metrics.append({"metrics": metrics_i, "pnl": pnl_i})

    logger.append(str(metrics))
    return metrics


if __name__ == '__main__':

    # =============== 1) backtest ===================

    open_prices_path_ = "../../../data/finance data/yahoo finance/open_prices.csv"
    word_count_matrix_path_ = '../../../data/word_count.p'
    index_matrix_path_ = '../../../data/index_matrix_labeled_yf_2020.p'

    logger_path_ = '../../../data/logs/'

    new_indices_path_ = None
    # new_indices_path_ = '../../../data/news_to_add_to_word_count_align_v1.p'

    weights_pickle_path_ = '../../../data/logs/weights/'
    # weights_pickle_path_ = 'data/logs/weights/'

    # index_matrix_with_sentiment_path_ = None
    index_matrix_with_sentiment_path_ = '../../../data/logs/sentiment/'

    alexandria = None
    # alexandria = "../../../data/alexandria/alexandria_index_matrix_joined.csv"
    # alexandria = "../../../data/alexandria/alexandria_index_matrix.csv"
    # alexandria = "../../../data/alexandria/alexandria_index_matrix_2020.csv"

    bert = None
    # bert = ("../../../data/BERT_index_matrix.csv", 'signal2_normalized')

    # domains_to_exclude = None
    domains_to_exclude = ['nbonews.com']

    weights_method_ = "EW"
    strategy_ = "LS"  # LS, L, S
    model_alpha_init_ = 50
    model_kappa_init_ = 0.985
    model_lambda_init_ = 0.5
    model_lambda_reg_init_ = 0
    to_rank = True
    CV_ = False
    expandingCV = False
    nr_companies_ = 20
    validation_periods_ = 5
    optimizer_ = 400
    trading_periods = 1
    start_date = '2016-09-01'
    end_date = '2020-02-27'
    fin_labels_type = 'normal'  # normal, beta_close, beta_adj_close
    refitting_frequency = 1
    recent_news_only = False

    run_backtest_wrapper(open_prices_path=open_prices_path_, word_count_matrix_path=word_count_matrix_path_,
                         index_matrix_path=index_matrix_path_, new_indices_path=new_indices_path_,
                         logger_path=logger_path_, weights_pickle_path=weights_pickle_path_,
                         index_matrix_with_sentiment_path=index_matrix_with_sentiment_path_,
                         weights_method=weights_method_, strategy=strategy_, validation_periods=validation_periods_,
                         CV=CV_, nr_companies=nr_companies_, model_alpha_init=model_alpha_init_,
                         model_kappa_init=model_kappa_init_, model_lambda_init=model_lambda_init_,
                         optimizer=optimizer_, expandingCV=expandingCV, alexandria=alexandria,
                         trading_periods=trading_periods, start_date=start_date, end_date=end_date,
                         fin_labels_type=fin_labels_type, to_rank=to_rank,
                         model_lambda_reg_init=model_lambda_reg_init_, refitting_frequency=refitting_frequency,
                         recent_news_only=recent_news_only, bert=bert)


    # =============== 2) figure 7 plots ===================
    # for list of some of old backtest runs see data/old_backtest_runs_list.txt

    # backtest_weights_df_ = weights_pickle_path_ + "EW_LS_2020-05-31_12-42-27_weights.p"  # SESTM, Weekly
    # backtest_weights_df_ = weights_pickle_path_ + "EW_LS_2020-05-31_12-50-39_weights.p"  # Alex, Weekly
    # backtest_weights_df_ = weights_pickle_path_ + "EW_LS_2020-04-16_09-12-25_weights.p"  # SESTM, Daily 9.30 AM
    # backtest_weights_df_ = weights_pickle_path_ + "EW_LS_2020-05-28_13-22-06_weights.p"  # Alex, Daily 9.30 AM
    # backtest_weights_df_ = weights_pickle_path_ + "EW_LS_2020-05-31_12-34-25_weights.p"  # SESTM, Daily 4.00 PM
    # backtest_weights_df_ = weights_pickle_path_ + "EW_LS_2020-05-31_12-32-03_weights.p"  # Alex, Daily 4.00 PM
    backtest_weights_df_ = weights_pickle_path_ + "EW_LS_2020-05-14_10-30-50_weights.p"  # Bert, Daily 9.30 AM (2018)

    spy_path_ = "../../../data/finance data/yahoo finance/spy.csv"

    # end_date = '2020-02-27'
    end_date = '2019-01-01'

    # path for saving daily returns series
    daily_returns_path = None
    # daily_returns_path = '../../../data/logs/daily_returns/daily_returns_2020/'

    # path for saving figure 7
    # figure_7_title = None
    # figure_7_title = "../../../paper pics/trading_results_added_random_seaborn.pdf"
    figure_7_title = "../../../paper pics/trading_results_BERT_seaborn.pdf"

    # random_pnls_path = None
    random_pnls_path = 'pnls_random_backtest'

    # backtest_plot_calculate_metrics(open_prices_path=open_prices_path_,
    #                                 backtest_weights_df=backtest_weights_df_,
    #                                 lag_start=-1, lag_end=1,
    #                                 spy_path=spy_path_, end_date=end_date, figure_8=False,
    #                                 daily_returns_path=daily_returns_path, figure_7=True,
    #                                 print_metrics=True, random_pnls_path=random_pnls_path,
    #                                 figure_7_title=figure_7_title)

    # =============== 3) backtest with random weights ===================

    strategy_ = "LS"
    nr_companies_ = 20
    trading_periods = 1
    start_date = '2016-09-01'
    end_date = '2020-02-27'

    # metrics_random = random_backtest(open_prices_path=open_prices_path_, strategy=strategy_,
    #                                  nr_companies=nr_companies_, trading_periods=trading_periods,
    #                                  start_date=start_date, end_date=end_date,
    #                                  weights_method=weights_method_, logger_path=logger_path_, N=500)
    #
    # pickle.dump(metrics_random, open("pnls_random_backtest", "wb"))







