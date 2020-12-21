import logging
import random
import numpy as np
import pandas as pd
import pickle
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import timedelta, datetime
from collections import defaultdict


def get_trading_weights_alexandria_bert(today, last, weights_method, strategy, nr_companies,
                                   columns, index_matrix, logger, recent_news_only):
    """
    Calculate trading weights based on the current news. Used in back-testing (see backtesting.py).

    :param today: day of trading
    :param last: previos trading day (for Monday, last trading day is Friday)
    :param weights_method: equal-weights (EW) or value-weights (VW)
    :param strategy: long-short (LS), long (L), short (S)
    :param nr_companies: number of companies to invest in
    :param columns: columns of dataframe with open-to-open returns, contains all tickers
    :param index_matrix: index matrix (TICKER, date_NYC, time_NYC, datetime_NYC, sentiment)
    :param logger: logging object
    :param recent_news_only: if true, for trading on day t consider news published between yesterday 4PM and today
        9AM. If false, for trading on day t consider news published between day t-1 9.30AM and today 9AM.
        Difference between yesterday and day t-1 is important for weekends and holidays.
    :return: weights for today, dimension: 1 x len(columns)
    """

    # select relevant news
    if recent_news_only:
        time_start = datetime.strptime("16:00:00", "%H:%M:%S").time()
    else:
        time_start = datetime.strptime("9:30:00", "%H:%M:%S").time()
    time_end = datetime.strptime("9:00:00", "%H:%M:%S").time()
    last_dt = datetime.combine(last, time_start)
    today_dt = datetime.combine(today, time_end)
    index_matrix = index_matrix[(index_matrix["datetime_NYC"] <= today_dt) & (index_matrix["datetime_NYC"] >= last_dt)]

    if len(index_matrix) > 0:
        pred_ticker_vector = list(zip(index_matrix.TICKER, index_matrix.sentiment))

        logger.append("Nr. of news for period from {} to {}: {}".format(last_dt, today_dt, len(pred_ticker_vector)))
        # logger.append("Predicted sentiment per news: {}".format(str(pred_ticker_vector)))

        # average sentiment for companies with multiple news
        pred_ticker_dict = defaultdict(list)
        for k, *v in pred_ticker_vector:
            pred_ticker_dict[k].append(v)
        pred_ticker_vector = list(map(lambda x: (x[0], np.mean(x[1])), list(pred_ticker_dict.items())))
        # logger.append("All companies : {}".format(str(pred_ticker_vector)))
        # pred_ticker_vector = list(map(lambda x: (x[0], sentiment_sharpe(x[1])), list(pred_ticker_dict.items())))
        tickers, sentiment = list(zip(*pred_ticker_vector))
        tickers, sentiment = list(tickers), np.array(list(sentiment))
        assert len(tickers) == len(set(tickers))  # check that there are no duplicate companies

        # find which companies to short/long
        argsort_sentiment = sentiment.argsort()
        long = []
        short = []
        if strategy == "L":
            long = argsort_sentiment[-nr_companies:]
        elif strategy == "S":
            short = argsort_sentiment[:nr_companies]
        else:
            if 2 * nr_companies > len(argsort_sentiment):
                nr_companies = int(len(argsort_sentiment) / 2)
            long = argsort_sentiment[-nr_companies:]
            short = argsort_sentiment[:nr_companies]

        long_companies = [tickers[i] for i in long]
        short_companies = [tickers[i] for i in short]
        logger.append("Long companies : {}".format(str(long_companies)))
        logger.append("Short companies : {}".format(str(short_companies)))

        # sentiment significance
        long_companies_sentiment = [sentiment[i] for i in long]
        short_companies_sentiment = [sentiment[i] for i in short]

        if "L" in strategy:
            logger.append("Long sentiment stats for {}: mean {}, std {}, max {}, min {}, count {}".format(today,
                          np.mean(long_companies_sentiment), np.std(long_companies_sentiment),
                          np.max(long_companies_sentiment), np.min(long_companies_sentiment),
                          len(long_companies_sentiment)))
        if "S" in strategy:
            logger.append("Short sentiment stats for {}: mean {}, std {}, max {}, min {}, count {}".format(today,
                          np.mean(short_companies_sentiment), np.std(short_companies_sentiment),
                          np.max(short_companies_sentiment), np.min(short_companies_sentiment),
                          len(short_companies_sentiment)))

        if weights_method == "EW":
            if strategy != "LS":
                if strategy == "L":
                    weight = 1 / len(long_companies)
                    weights_today = {company: (weight if company in long_companies else 0) for company in columns}
                else:  # S
                    weight = 1 / len(short_companies)
                    weights_today = {company: (weight if company in short_companies else 0) for company in columns}
                # assert sum(weights_today.values()) == 1
            else:  # LS
                assert len(long_companies) == len(short_companies)
                # weight = 1 / (2*len(long_companies))
                weight = 1 / len(long_companies)
                weights_today = dict()
                for company in columns:
                    if company in long_companies:
                        weights_today[company] = weight
                    elif company in short_companies:
                        weights_today[company] = -weight
                    else:
                        weights_today[company] = 0
        # TODO: implement value weights, for this need to first get market capitalization data
        else:  # VW
            pass

    else:  # no news, do not invest
        weights_today = {company: 0 for company in columns}

    return weights_today


def get_trading_weights_random(strategy, nr_companies, columns, logger):
    """
    Get random trading weights. Serves as a baseline for LS strategies.

    :param weights_method:
    :param strategy:
    :param nr_companies:
    :param columns:
    :param logger:
    :param seed: for reproducibility
    :return:
    """

    # pick random companies
    if strategy == "LS":
        long = random.sample(list(range(len(columns))), nr_companies)
        short = random.sample([x for x in list(range(len(columns))) if x not in long], nr_companies)
        assert len(set(long).intersection(set(short))) == 0, "Going long and short simultaneously for same company!"
    elif strategy == "L":
        long = random.sample(list(range(len(columns))), nr_companies)
    elif strategy == "S":
        short = random.sample(list(range(len(columns))), nr_companies)

    # recover tickers
    long_companies = [columns[i] for i in long]
    short_companies = [columns[i] for i in short]
    # logger.append("Long companies : {}".format(str(long_companies)))
    # logger.append("Short companies : {}".format(str(short_companies)))

    # calculate weights
    if strategy != "LS":
        if strategy == "L":
            weight = 1 / len(long_companies)
            weights_today = {company: (weight if company in long_companies else 0) for company in columns}
        else:  # S
            weight = 1 / len(short_companies)
            weights_today = {company: (weight if company in short_companies else 0) for company in columns}
    else:  # LS
        assert len(long_companies) == len(short_companies)
        # weight = 1 / (2*len(long_companies))
        weight = 1 / len(long_companies)
        weights_today = dict()
        for company in columns:
            if company in long_companies:
                weights_today[company] = weight
            elif company in short_companies:
                weights_today[company] = -weight
            else:
                weights_today[company] = 0
    return weights_today


def get_business_day(date):
    cal = USFederalHolidayCalendar()
    while date.isoweekday() > 5 or date in cal.holidays():
        date += timedelta(days=1)
    return date


def pnl_metrics(pnl, spy_returns=None, n=252, print_=True):
    # Average daily return, confidence interval
    daily_returns_mean = pnl.mean()
    daily_returns_std = pnl.std()
    conf_int = (round((daily_returns_mean-1.96*daily_returns_std/np.sqrt(len(pnl)))*10000, 2),
                round((daily_returns_mean+1.96*daily_returns_std/np.sqrt(len(pnl)))*10000, 2))

    # Max draw-down
    md = pnl.cumsum()
    roll_max = md.rolling(window=md.shape[0], min_periods=1).max()
    daily_drawdown = md - roll_max
    mdd = -daily_drawdown.min()

    # Return
    ret = pnl.cumsum().iloc[-1]

    # Annualized return
    pnl_shape = pnl.cumsum().shape[0]
    # ann_return = ((1 + pnl.cumsum().iloc[-1]) ** (n / pnl_shape) - 1)*100
    # ann_return = ((1 + daily_returns_mean)**252-1)*100
    ann_return = daily_returns_mean * n * 100

    # Annualized Sharpe
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(n)

    # Annualized volatility
    ann_vol = daily_returns_std * np.sqrt(n)*100

    if print_:
        print("Annualized Sharpe: {}".format(round(sharpe, 2)))
        print("Annualized return: {}".format(round(ann_return, 2)))
        print("Return: {}".format(round(ret, 2)))
        print("MDD: {}".format(round(mdd, 6)))
        print("Average daily return [bps]: {} ".format(round(daily_returns_mean*10000, 2)))
        print("Average daily return - confidence interval [bps]: {}".format(str(conf_int)))

        # excess of sp500
        if spy_returns is not None:
            print("Average excess return [bps]: {}".format(round((pnl - spy_returns.Open).mean()*10000, 2)))

    return sharpe, daily_returns_mean*10000, daily_returns_std*10000, mdd, ann_return, ret, conf_int, ann_vol


def random_strategy_pnls(path, end_date="2020-02-27"):
    """
    Read in pnls from random strategy.

    """
    pnls = pickle.load(open(path, "rb"))
    df = pd.DataFrame()
    for i in range(len(pnls)):
        df[i] = pnls[i]['pnl'].cumsum()
    df = df.loc[:end_date]
    return df


class Logger(object):
    def __init__(self, log_dir, weights_method, strategy, dt):
        self.name = log_dir + weights_method + "_" + strategy + "_" + "backtest_" + dt.strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(filename=self.name, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

    @staticmethod
    def append(x, *args):
        if len(args) == 1:
            s = "{:<40}{:<15}".format(x, str(args[0]))
        elif len(args) == 2:
            s = "{:<40}{:<15}{:<15}".format(x, str(args[0]), str(args[1]))
        elif len(args) == 3:
            s = "{:<40}{:<15}{:<15}{:<15}".format(x, str(args[0]), str(args[1]), str(args[2]))
        else:
            s = "{:<40}".format(x)

        logging.info(s)
        print(s)

    @staticmethod
    def create_log():
        logging.shutdown()