"""
Utils functions to compute yahoo finance labels.
"""

import yfinance as yf
import pickle
import pandas as pd
import numpy as np
from src.data.utils.keywordsIO import load_company_to_ticker_dict


def get_labels_yf(company_to_tickers_path, labels_yf_path):
    """
    Get financial labels based on Yahoo Finance adjusted close returns.

    :param company_to_tickers_path:
    :param labels_yf_path:
    """
    company_ticker_dict = load_company_to_ticker_dict(company_to_tickers_path)
    TICKERS = list(company_ticker_dict.values())
    print("Nr. of tickers: {}".format(len(TICKERS)))

    # download available tickers from yahoo finance
    data = yf.download(" ".join(TICKERS), start="2016-09-02", end="2019-11-14")
    adj_close_df = data["Adj Close"]

    # calculate labels
    labels_df = (adj_close_df.shift(-1) - adj_close_df.shift(2)) / adj_close_df.shift(2)

    # Add labels for weekends and holidays.
    # Importance of limit parameter in .fillna()!!
    # (without it a company that gets on the exchange in 2018 will get labels for all the news back to 2016...)
    date_range = pd.date_range(start="2016-09-01", end="2019-11-13")
    labels_df = labels_df.reindex(date_range).rename_axis('date').reset_index().fillna(method='bfill', limit=3)

    # prepare labels dataframe for join with news, stack columns on top of one another
    n = len(labels_df)
    labels_df_final = pd.DataFrame(columns=["date", "LABEL", "TICKER"], data=np.zeros(shape=(n * 573, 3)))
    cols = list(labels_df.columns)
    for i in range(len(cols)):
        ticker = cols[i]
        if ticker != "date":
            tmp = labels_df[["date", ticker]]
            tmp["TICKER"] = [ticker] * n
            # tried concatenating dataframes with pandas append, but was super slow...
            labels_df_final.loc[(i - 1) * n:i * n - 1, :] = tmp.values

    print(labels_df.head())
    print(labels_df.shape)

    labels_df_final.to_csv(labels_yf_path)


def yahoo_finance_misc(company_to_tickers_path, open_path_csv, close_path_csv, spy_path_csv):
    """
    Get miscellaneous data from Yahoo finance.
    - open prices data: used in backtesting
    - adjusted close prices data: used for absorption analysis
    - sp500 index data: used as a baseline index

    :param company_to_tickers_path:
    :param open_path_csv:
    :param close_path_csv:
    :param spy_path_csv:
    """

    company_ticker_dict = load_company_to_ticker_dict(company_to_tickers_path)
    TICKERS = list(company_ticker_dict.values())
    print("Nr. of tickers: {}".format(len(TICKERS)))

    # download available tickers from yahoo finance, as well as sp500
    data = yf.download(" ".join(TICKERS), start="2016-09-02", end="2020-03-31")
    spy = yf.download("SPY", start="2016-09-02", end="2020-03-31")

    # open prices data
    open_df = data["Open"]
    open_df['date'] = open_df.index
    open_df.reset_index(drop=True, inplace=True)
    open_df.to_csv(open_path_csv)

    # adjusted close prices data
    adj_close_df = data["Adj Close"]
    adj_close_df['date'] = adj_close_df.index
    adj_close_df.reset_index(drop=True, inplace=True)
    adj_close_df.to_csv(close_path_csv)

    # sp500
    spy.to_csv(spy_path_csv)


if __name__ == "__main__":

    company_to_tickers_path_= "../../../data/sp500_list_25-02-2020.txt"
    open_path_csv_ = "../../../data/finance data/yahoo finance/test/open_prices.csv"
    close_path_csv_ = "../../../data/finance data/yahoo finance/test/adj_close_prices.csv"
    spy_path_csv_ = "../../../data/finance data/yahoo finance/test/spy.csv"

    yahoo_finance_misc(company_to_tickers_path_, open_path_csv_, close_path_csv_, spy_path_csv_)
