"""
Price-based event study.
Comparing intraday effects of CommonCrawl and Alexandria datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time as time_lib
import pickle
import statsmodels.api as sm

# ugly hack cause normal import causes error when using in jupyter notebook.
# when running locally comment lines 17-19 and uncomment line 20
import sys
sys.path.append("../../src/models/benchmark/utils")
from event_study_utils import *
# from src.models.benchmark.utils.event_study_utils import *


def calculate_binned_returns(ticker, nbbo_path, save_path, years=[2017, 2018, 2019], bin_size_minutes=10, lag=60):
    """
    Calculate binned returns for price-based event study analysis.

    :param ticker:
    :param nbbo_path: paht to NBBO daily files
    :param save_path: where to save parsed returns
    :param years: which years to consider
    :param bin_size_minutes: size of bin (in  minutes)
    :param lag: how many past days to consider when calculating beta-adjusted returns

    :return normal binned returns, abnormal binned returns (beta adjusted)
    """
    # set up binning times
    times = binning_times(bin_size_minutes)

    # import SPX data
    spx = read_SPX_data()
    spx["date"] = spx.index.date
    spx["minute"] = spx.index.time
    # ugly solution for the fact that we have returns only until 15:59 in SPX dataset
    spx["minute"] = spx["minute"].apply(lambda x: datetime.time(16, 0, 0) if x == datetime.time(15, 59, 0) else x)
    spx.reset_index(drop=True, inplace=True)

    # import nbbo files
    # TODO: this part is way too slow. For every company all the daily files are read separately. Fix this.
    start = time_lib.time()

    files = []
    for year in years:
        path = nbbo_path + str(year)
        for r, d, f in os.walk(path):
            for file in f:
                if str(year) in file:
                    files.append(os.path.join(r, file))

    nbbo = pd.DataFrame()
    for file in files:
        # read in the file
        nbbo_daily_df = pd.read_csv(file, usecols=["date", "minute", "last_bid", "last_ask", "sym_root"])

        # filter on price timestamps and ticker
        nbbo_daily_df = nbbo_daily_df[nbbo_daily_df["sym_root"] == ticker]
        nbbo_daily_df["minute"] = nbbo_daily_df["minute"].apply(lambda x: parse_time_fast(x[7:15]))
        nbbo_daily_df = nbbo_daily_df[nbbo_daily_df["minute"].isin(times)]

        # calculate mid-quote close price
        nbbo_daily_df["close"] = (nbbo_daily_df["last_bid"] + nbbo_daily_df["last_ask"]) / 2
        nbbo_daily_df = nbbo_daily_df[["date", "minute", "close"]]

        nbbo = nbbo.append(nbbo_daily_df)

    end = time_lib.time()
    print("Importing and processing NBBO files took: {}".format(end - start))

    # join nbbo and spx (left join on date and minute)
    start = time_lib.time()

    nbbo["date"] = nbbo["date"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())
    nbbo.reset_index(drop=True, inplace=True)

    merged_df = pd.merge(nbbo, spx, how="left", on=["date", "minute"])
    del spx
    del nbbo

    merged_df["return_{}".format(ticker)] = merged_df.groupby("date")["close_x"].pct_change()
    merged_df["return_SPX"] = merged_df.groupby("date")["close_y"].pct_change()

    merged_df.dropna(inplace=True)

    end = time_lib.time()
    print("Join and processing of joined dataframe took: {}".format(end - start))

    # run linear regressions
    start = time_lib.time()

    merged_df["datetime"] = merged_df.apply(lambda r: pd.datetime.combine(r['date'], r['minute']), 1)
    merged_df.set_index("datetime", drop=True, inplace=True)

    merged_df["alpha"] = None
    merged_df["beta"] = None
    merged_df["alpha_pvalue"] = None
    merged_df["beta_pvalue"] = None

    dates = list(merged_df.date.unique())
    for i in range(lag, len(dates)):
        start_date = dates[i - lag]
        end_date = dates[i]
        slice_df = merged_df[(merged_df["date"] >= start_date) & (merged_df["date"] < end_date)].copy()
        # regress company returns on SPX returns
        fit = sm.OLS(slice_df.loc[:, "return_{}".format(ticker)].values,
                     sm.add_constant(slice_df.loc[:, "return_SPX"].values)).fit()
        index_rows = merged_df[merged_df["date"] == end_date].index
        merged_df.loc[index_rows, "alpha"] = fit.params[0]
        merged_df.loc[index_rows, "beta"] = fit.params[1]
        merged_df.loc[index_rows, "alpha_pvalue"] = fit.pvalues[0]
        merged_df.loc[index_rows, "beta_pvalue"] = fit.pvalues[1]

    merged_df.dropna(inplace=True)
    end = time_lib.time()
    print("Linear regressions took: {}".format(end - start))

    # calculate abnormal returns
    merged_df["abnormal_return_{}".format(ticker)] = merged_df["return_{}".format(ticker)] - \
                                                     (merged_df["alpha"] + merged_df["beta"] * merged_df["return_SPX"])

    # pivot tables
    merged_df["abnormal_return_{}".format(ticker)] = merged_df["abnormal_return_{}".format(ticker)].astype(float)

    abnormal_returns_df = pd.pivot_table(values="abnormal_return_{}".format(ticker), columns="minute", index="date",
                                         data=merged_df)
    normal_returns_df = pd.pivot_table(values="return_{}".format(ticker), columns="minute", index="date",
                                       data=merged_df)

    # save parsed dataframes
    abnormal_returns_df.to_csv(save_path + "/abnormal_returns/{}_{}.csv".format(ticker, bin_size_minutes))
    normal_returns_df.to_csv(save_path + "/normal_returns/{}_{}.csv".format(ticker, bin_size_minutes))

    return


def select_news(ticker, news_df, t1, t2, bin_size_min, trading_dates):
    """
    Select relevant news for ticker.

    :param ticker:
    :param news_df:
    :param t1:
    :param t2:
    :param bin_size_min:
    :param trading_dates:
    :return:
    """

    market_open = datetime.time(9, 30, 0)
    market_close = datetime.time(16, 0, 0)
    start = (datetime.datetime.combine(datetime.date(1, 1, 1), market_open) + datetime.timedelta(
        minutes=t1 * bin_size_min)).time()
    end = (datetime.datetime.combine(datetime.date(1, 1, 1), market_close) - datetime.timedelta(
        minutes=t2 * bin_size_min)).time()

    times = binning_times(bin_size_min)

    news = news_df[
        (news_df["TICKER"] == ticker) & (news_df["pred_sentiment"].notnull()) & (news_df["time_NYC"] > start) &
        (news_df["time_NYC"] <= end) & (news_df["date_NYC"].isin(trading_dates))]

    news["bin"] = news["time_NYC"].apply(lambda x: binning(x, times))

    for i in range(-t1, t2 + 1):
        news["Bin{}".format(i)] = np.nan

    news.reset_index(drop=True, inplace=True)

    return news


def align(news_df, binned_returns_df, t1, t2):
    """
    Align binned returns and news.

    :param news_df:
    :param binned_returns_df:
    :param t1:
    :param t2:
    :return:
    """

    # align
    for i in range(len(news_df)):
        bin = int(news_df.loc[i]["bin"])
        date = news_df.loc[i]["date_NYC"]
        aligned_returns = binned_returns_df.loc[date][bin - t1:bin + t2 + 1]
        news_df.iloc[i, 5:] = list(aligned_returns)

    # if there was no return available, set it to 0
    news_df.fillna(0, inplace=True)

    return news_df


def plot_event_study(aligned_df_cc, aligned_df_alex, ticker, t1):
    """
    Plot event study plots for both Alexandria and CC datasets.

    :param aligned_df_cc:
    :param aligned_df_alex:
    :param ticker:
    :param t1:
    :return:
    """
    # TODO: make main title appear nicer. Also figure out optimal
    #   composition for all 8 plots.

    positive_news_cc = aligned_df_cc[aligned_df_cc["pred_sentiment"] > 0.50]
    negative_news_cc = aligned_df_cc[aligned_df_cc["pred_sentiment"] < 0.50]
    neutral_news_cc = aligned_df_cc[aligned_df_cc["pred_sentiment"] == 0.5]

    print("{} number positive news {}".format("CC", len(positive_news_cc)))
    print("{} number negative news {}".format("CC", len(negative_news_cc)))
    print("{} number neutral news {}".format("CC", len(neutral_news_cc)))

    positive_news_alex = aligned_df_alex[aligned_df_alex["pred_sentiment"] > 0.50]
    negative_news_alex = aligned_df_alex[aligned_df_alex["pred_sentiment"] < 0.50]
    neutral_news_alex = aligned_df_alex[aligned_df_alex["pred_sentiment"] == 0.5]

    print("{} number positive news {}".format("Alexandria", len(positive_news_alex)))
    print("{} number negative news {}".format("Alexandria", len(negative_news_alex)))
    print("{} number neutral news {}".format("Alexandria", len(neutral_news_alex)))

    def plot_fun(x, sent, index_df_type, ax1, ax2, bincenter=t1,):
        x.drop(['date_NYC', 'time_NYC', 'TICKER', 'pred_sentiment', 'bin'], axis=1, inplace=True)

        # ax1.ticklabel_format(axis='y', style='sci')
        x.mean().plot(color="blue", ax=ax1)
        (x.mean() + x.sem()).plot(color="blue", style="--", ax=ax1)
        (x.mean() - x.sem()).plot(color="blue", style="--", ax=ax1)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.axvline(x=bincenter, color='r', linestyle='--')
        ax1.set_title("Avg. return. {} news. {}".format(sent, index_df_type))

        # ax2.ticklabel_format(axis='y', style='sci')
        x.cumsum(axis=1).mean().plot(ax=ax2)
        ax2.axvline(x=bincenter, color='r', linestyle='--')
        ax2.set_title("Cum. return. {} news. {}".format(sent, index_df_type))

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 10))
    fig.suptitle(ticker, fontsize=32)

    plot_fun(positive_news_cc, "Positive", "CC", ax1=axes[0, 0], ax2=axes[1, 0])
    plot_fun(negative_news_cc, "Negative", "CC", ax1=axes[0, 2], ax2=axes[1, 2])
    plot_fun(positive_news_alex, "Positive", "ALEX", ax1=axes[0, 1], ax2=axes[1, 1])
    plot_fun(negative_news_alex, "Negative", "ALEX", ax1=axes[0, 3], ax2=axes[1, 3])

    # fig.tight_layout()
    plt.show()

    return


def event_study(ticker, returns_path, bin_size_minutes, t1, t2, alex_df, cc_df):
    """
    Main function for price based event study.

    :param ticker:
    :param returns_path: with this can control what type of returns are use, either 'normal' or
        'abnormal' (beta-adjusted)
    :param bin_size_minutes:
    :param t1:
    :param t2:
    :param alex_df:
    :param cc_df:
    :return:
    """

    # read in prebinned returns
    returns_df = pd.read_csv(returns_path + '/{}_{}.csv'.format(ticker, bin_size_minutes),
                             parse_dates=['date'], index_col='date')

    # read in news
    if ticker == "GOOG":
        ticker = "GOOGL"
    news_alex = select_news(ticker, alex_df, t1, t2, bin_size_minutes, list(returns_df.index))
    news_cc = select_news(ticker, cc_df, t1, t2, bin_size_minutes, list(returns_df.index))

    # align news and prebinned returns
    news_alex = align(news_alex, returns_df, t1, t2)
    news_cc = align(news_cc, returns_df, t1, t2)

    # plot
    plot_event_study(aligned_df_alex=news_alex, aligned_df_cc=news_cc, ticker=ticker, t1=t1)


if __name__ == '__main__':

    # =================== 1) Calculating binned returns ==========================
    # nbbo_path_ = '../../../data/finance data/minute OHLC/NBBO/'
    # save_path_ = '../../../data/finance data/event_study_parsed'
    # # tickers = ['AMZN', 'AAPL', 'GOOG', 'MSFT', 'BA', 'NFLX', 'WMT', 'DIS', 'INTC', 'WFC', 'CBS', 'GS']
    # tickers = ['GOOG']
    #
    # for ticker in tickers:
    #     print("Parsing returns for {}.".format(ticker))
    #     calculate_binned_returns(ticker, nbbo_path_, save_path_)
    #     print("Parsing returns for {} done!".format(ticker))

    # ================= 2) Selecting relevant news ===============================

    # specify event study params
    alex_path = "../../../data/alexandria/alexandria_index_matrix.csv"
    cc_path = "../../../data/index_matrices_2020/EW_LS_2020-04-02_10-38-26_index_matrix.p"
    returns_path = '../../../data/finance data/event_study_parsed/normal_returns'
    T1 = T2 = 12
    bin_size_minutes = 10

    # read in index matrices for both Alexandria and CC
    alex_df = pd.read_csv(alex_path, index_col=0, parse_dates=["date_NYC"])
    alex_df["time_NYC"] = alex_df["time_NYC"].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time())
    alex_df.rename({"sentiment": "pred_sentiment"}, inplace=True, axis=1)
    alex_df = alex_df[['date_NYC', 'time_NYC', 'TICKER', 'pred_sentiment']]

    cc_df = pickle.load(open(cc_path, "rb"))
    cc_df['date_NYC'] = pd.to_datetime(cc_df['date_NYC'])
    cc_df = cc_df[['date_NYC', 'time_NYC', 'TICKER', 'pred_sentiment']]

    # run event study
    event_study(ticker='AMZN', returns_path=returns_path, bin_size_minutes=bin_size_minutes,
                t1=T1, t2=T2, alex_df=alex_df, cc_df=cc_df)

