"""
Alignment of news and finance data.
"""
import pandas as pd
import numpy as np
import pickle
from collections import Counter # used in eval !
import dateutil
import datetime
import pytz
import time
from functools import partial
from scipy.sparse import vstack
from src.data.utils.keywordsIO import *
import urllib.parse as uparse


def add_news_to_word_count(word_count_mat, new_indices):
    """
    In case of multiple alignment (align_v1), new rows have to be added to word count matrix.
    Only used with 2019 data.
    :param word_count_mat: word_count matrix (scipy sparse matrix)
    :param new_indices: indices of news to be added (tuple of pandas index objects)
    :return: word_count matrix with added new rows
    """

    print("Dimension of word count matrix at the start: {}".format(word_count_mat.shape[0]))

    new_indices = [list(x) for x in new_indices]
    for ind_list in new_indices:
        df_to_add = word_count_mat[ind_list, :].copy()
        word_count_mat = vstack([word_count_mat, df_to_add])
        print("Nr. of added rows: {}".format(df_to_add.shape[0]))

    return word_count_mat


def datetime_parse_financial_labels(x):
    """
    Does datetime parsing for financial labels alignment. If the article is published before 4pm on a given day,
    date is left unchanged. If article is published after 4pm we increase date for one day.
    Note that there is no need to separately consider weekends and holidays
    as they get aligned with the first next trading day anyway.
    See last sentence on page 12 of the benchmark paper.
    :param x:
    :return:
    """
    if x.date_NYC and x.time_NYC:
        _4pm = datetime.time(16, 0, 0)
        if x.time_NYC > _4pm:  # if news published after 4pm, increase date for one day
            date_NYC_LABEL = x.date_NYC + datetime.timedelta(days=1)
        else:
            date_NYC_LABEL = x.date_NYC
    else:
        date_NYC_LABEL = None
    return date_NYC_LABEL


def training_labels_alignment_2020(index_matrix, labels_path,
                                   urls_path, company_to_tickers_path, beta_adjusted_labels_path=None):
    """
    - parse company name to ticker
    - convert UTC tz to NYC tc
    - financial labels datetime shift (Kelly paper), see function datetime_parse_financial_labels
    - join with financial labels (normal, beta-adjusted)
    - add domains
    :param index_matrix:
    :param labels_path:
    :param urls_path:
    :param company_to_tickers_path:
    :param beta_adjusted_labels_path:
    :return:
    """

    # convert company names to tickers
    company_ticker_dict = load_company_to_ticker_dict(company_to_tickers_path)
    index_matrix["TICKER"] = index_matrix["company"].map(lambda x: company_ticker_dict[x])
    print("Shape of index matrix before left join {}".format(index_matrix.shape))
    print("Nr. of news without ticker alignment: {}".format(index_matrix.TICKER.isna().sum()))

    # datetime, convert to NYC timezone
    start = time.time()
    index_matrix['publish_date_NYC'] = pd.DatetimeIndex(pd.to_datetime(index_matrix['publish_date_utc'],
                                                                       errors='coerce')).tz_convert('America/New_York')

    def parse_date(x):
        try:
            return x.date()
        except ValueError:
            return None

    def parse_time(x):
        try:
            return x.time()
        except ValueError:
            return None

    def combine_date_time(x):  # could be omitted and we would simply use publish_date_NYC instead
        if x.date_NYC and x.time_NYC:
            return datetime.datetime.combine(x.date_NYC, x.time_NYC)
        else:
            return None

    index_matrix['date_NYC'] = index_matrix['publish_date_NYC'].apply(lambda x: parse_date(x))
    index_matrix['time_NYC'] = index_matrix['publish_date_NYC'].apply(lambda x: parse_time(x))
    index_matrix["date_NYC_LABEL"] = index_matrix.apply(datetime_parse_financial_labels, axis=1)
    index_matrix["datetime_NYC"] = index_matrix.apply(combine_date_time, axis=1)
    end = time.time()
    print('Datetime parsing took {} seconds!'.format(round(end-start, 2)))

    # import normal labels, join
    df_labels = pd.read_csv(labels_path, parse_dates=["date"])
    df_labels = df_labels[["date", "TICKER", "LABEL"]]
    df_labels["date"] = df_labels["date"].map(lambda x: x.date())
    df_labels.drop_duplicates(subset=["date", "TICKER"], inplace=True)  # that was the reason for mismatch in nr. of row

    new_df = pd.merge(index_matrix, df_labels, how="left", left_on=["date_NYC_LABEL", "TICKER"],
                      right_on=["date", "TICKER"], indicator=True, validate="many_to_one")
    new_df = new_df[["TICKER", "date_NYC", "time_NYC", "LABEL", "date_NYC_LABEL", "datetime_NYC"]]

    assert len(new_df) == len(index_matrix)  # sanity check for join

    # import beta adjusted labels, join
    if beta_adjusted_labels_path is not None:
        df_labels = pd.read_csv(beta_adjusted_labels_path, index_col=0, parse_dates=['date'])
        df_labels.drop_duplicates(subset=["date", "TICKER"],
                                  inplace=True)  # that was the reason for mismatch in nr. of rows
        df_labels.rename(columns={'Abnormal_Close': 'beta_labels_close',
                                  'Abnormal_Adj Close': 'beta_labels_adjusted_close'}, inplace='True')
        df_labels["date"] = df_labels["date"].map(lambda x: x.date())

        new_df = pd.merge(new_df, df_labels, how="left", left_on=["date_NYC_LABEL", "TICKER"],
                          right_on=["date", "TICKER"], indicator=True, validate="many_to_one")
        new_df = new_df[["TICKER", "date_NYC", "time_NYC", 'datetime_NYC',
                         "LABEL", 'beta_labels_close', 'beta_labels_adjusted_close']]

        assert len(new_df) == len(index_matrix)  # sanity check for join

    # add domains
    df_urls = pd.read_csv(urls_path, usecols=['title', 'publish_date_utc', 'url',  'company'])
    new_df['url'] = df_urls['url']
    new_df['url'] = new_df['url'].apply(lambda x: uparse.urlparse(x).netloc if type(x) is str else None)

    return new_df


if __name__ == "__main__":
    from src.models.utils.large_pickle_io import read_large_pickle, write_large_pickle

    ###################### training_labels_alignment #####################
    company_to_tickers_path_ = "../../../../data/sp500_list_25-02-2020.txt"
    labeled_df_path_ = "../../../../data/finance data/yahoo finance/adj_close_labels.csv"
    domains_path = '../../../../data/alignment_output.csv'
    word_matrix_path = '../../../../data/word_count.p'

    word_matrix = read_large_pickle(word_matrix_path)
    index_matrix_ = word_matrix["index"]
    print(list(index_matrix_.columns))
    del word_matrix

    joined_df = training_labels_alignment_2020(index_matrix_, labeled_df_path_,
                                               domains_path, company_to_tickers_path_)

    print(joined_df.head(10))

    write_large_pickle(joined_df, "../../../../data/index_matrix_labeled_yf_2020_.p")
