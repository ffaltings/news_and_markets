"""
Utils functions to compute WRDS finance labels.
"""

import pandas as pd
import numpy as np


def get_labels(ticker, crsp_path, date_range, log_returns=False):
    """
    Helper function for precomputing labeled dataframe.

    :param ticker:
    :param crsp_path: path to CRSP.csv file with daily stock returns (can be found on Polybox)
    :param date_range:
    :param log_returns:
    :return: labeled dataframe
    """
    df_crsp = pd.read_csv(crsp_path, parse_dates=["date"], index_col=0)
    assert ticker in list(df_crsp.TICKER.unique()), "TICKER not found in price data!"

    df_crsp = df_crsp[df_crsp["TICKER"] == ticker]

    # calculate returns between t-2 close price and t+1 close price
    if not log_returns:
        df_crsp["LABEL"] = (df_crsp.PRC.shift(-1) - df_crsp.PRC.shift(2)) / df_crsp.PRC.shift(2)
    else:
        df_crsp["LABEL"] = np.log(df_crsp.PRC.shift(-1) / df_crsp.PRC.shift(2))

    # weekend labels
    df_crsp = df_crsp.set_index('date').reindex(date_range).rename_axis('date').reset_index().fillna(method='bfill')
    return df_crsp


def CRSP_merge(CRSP_2016_2018_path, CRSP_2019_path, CRSP_path):
    """
    Daily returns WRDS on WRDS are available only until end of 2018 (CRSP).
    Daily returns WRDS for 2019 are calculated from minute OHLC data that we derived from TAQ dataset.
    This function merges the two.

    :param CRSP_2016_2018_path: file was obtained via web query:
            https://wrds-web.wharton.upenn.edu/wrds/ds/crsp/stock_a/dsf.cfm?navId=128
            Fields retrieved: date, ticker, company name, (close) price, open price
    :param CRSP_2019_path: file derived from minute OHLC data using ohlc_to_daily function
    :param CRSP_path: path to save merged dataframe
    :return: merged dataframe
    """
    CRSP_2016_2018 = pd.read_csv(CRSP_2016_2018_path, parse_dates=["date"], index_col=0)
    CRSP_2019 = pd.read_csv(CRSP_2019_path, parse_dates=["date"], index_col=0)
    CRSP_2016_2018 = CRSP_2016_2018[["date", "TICKER", "PRC", "OPENPRC"]]
    CRSP = CRSP_2016_2018.append(CRSP_2019, ignore_index=True)
    CRSP.sort_values(by=["TICKER", "date"], inplace=True)
    CRSP.drop_duplicates(subset=["date", "TICKER"], inplace=True)
    CRSP.to_csv(CRSP_path)
    return CRSP


def ohlc_to_daily(file):
    """
    Calculate daily open and close price from minute OHLC data (2019).

    :param file: directory contatining 2019 daily minuteOHLC files
    :return:
    """
    daily_df = pd.DataFrame(columns=["date", "TICKER", "PRC","OPENPRC"])
    df_ohlc = pd.read_csv(file, index_col=0, parse_dates=["date"])
    if len(df_ohlc) == 0:
        print("DROPPED ", file)
    print("Processing ", file)
    tickers = list(df_ohlc.sym_root.unique())
    for ticker in tickers:
        try:
            tmp = df_ohlc[(df_ohlc["sym_root"] == ticker) & (df_ohlc["minute"] >= "0 days 09:30:00.000000000") &  \
                          (df_ohlc["minute"] <= "0 days 16:00:00.000000000")]
            tmp_first = tmp.iloc[0,:]
            tmp_last = tmp.iloc[-1,:]
            tmp_dict = {"date": tmp_first["date"],"TICKER": tmp_first["sym_root"], "PRC": tmp_last["last"], \
                                "OPENPRC":tmp_first["open"]}
        except Exception as error:
            print("Ticker {} in file {} had error: {}".format(ticker, file, error))
            tmp_dict = {"date": None, "TICKER": None, "PRC":None, \
                        "OPENPRC": None}
        daily_df.loc[len(daily_df)] = tmp_dict
    return daily_df


def open_returns_backtest(CRSP_path):
    """

    :param CRSP_path: path to CRSP file with daily returns WRDS for each company for each day
    :return: dataframe with open returns (nr_days x nr_companies)
    """
    df_crsp = pd.read_csv(CRSP_path, parse_dates=["date"], index_col=0)
    TICKERS_CRSP = list(df_crsp.TICKER.unique())

    # get all trading dates, since AMZN is traded a lot we take its index
    dt = df_crsp[df_crsp["TICKER"] == "AMZN"].date
    df_open = pd.DataFrame(index=dt, columns=TICKERS_CRSP)

    for ticker in TICKERS_CRSP:
        df_open[ticker] = df_crsp[df_crsp["TICKER"] == ticker].set_index('date').reindex(dt). \
                                                rename_axis('date').OPENPRC
    return df_open


if __name__ == "__main__":

    FINANCE_PREPROCESS = False
    OPEN_DF = True

    if FINANCE_PREPROCESS:

        import multiprocessing as mp
        import os

        pool = mp.Pool(processes=os.cpu_count())

        print(os.cpu_count())

        path = '../../../../data/finance data/minute OHLC'

        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if "2019" in file:
                    files.append(os.path.join(r, file))

        print(len(files))

        dfs = pool.map(ohlc_to_daily, files)

        df_ohlc_ = pd.DataFrame(columns=["date", "TICKER", "PRC", "OPENPRC"])
        for df_ in dfs:
            df_ohlc_ = df_ohlc_.append(df_, ignore_index=True)
        df_ohlc_.sort_values(by=["TICKER", "date"])

        df_ohlc_.to_csv("../../../../data/finance data/daily returns WRDS/CRSP_2019.csv")

        del df_ohlc_

        crsp = CRSP_merge("../../../../data/finance data/daily returns WRDS/CRSP_2016_2018.csv",
                            "../../../../data/finance data/daily returns WRDS/CRSP_2019.csv",
                          "../../../../data/finance data/daily returns WRDS/CRSP.csv")

        date_range_ = pd.date_range(start=crsp.date.min(), end=crsp.date.max())
        TICKERS_CRSP = list(crsp.TICKER.unique())

        del crsp

        df_labels = pd.DataFrame(columns=["date", "TICKER", "PRC", "OPENPRC", "LABEL"])
        for ticker in TICKERS_CRSP:
            tmp = get_labels(ticker, "../../../../data/finance data/daily returns WRDS/CRSP.csv", date_range_)
            df_labels = df_labels.append(tmp, ignore_index=True)

        df_labels.to_csv("../../../../data/finance data/daily returns WRDS/CRSP_labels.csv")

    if OPEN_DF:
        open_df = open_returns_backtest("../../../../data/finance data/daily returns WRDS/CRSP.csv")
        open_df.to_csv("../../../../data/finance data/daily returns WRDS/CRSP_open_new.csv")