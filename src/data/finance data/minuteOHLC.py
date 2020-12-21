"""
Script that was run on WRDS cluster to obtain minute OHLC data for sp500 companies.
Note that script was run separately for each year (by setting year_def and year_ param below),
since there is a limited amount of space available for storage on the cluster
(did not want to make use of NYU's shared scratch storage).
"""

import wrds
import pandas as pd
from multiprocessing import Pool
import os
import timeit

TICKERS = []
with open("TICKERS.txt", "r") as file:
    line = file.readline()
    while line:
        TICKERS.append(line.strip())
        line = file.readline()
    file.close()
tickers_def = {"tickers": tuple(TICKERS)}

year_def = 2017
type_def = "nbbom"


def process_daily_taq(file_name, tickers=tickers_def, year=year_def, type=type_def):
    """
    Process file with all daily trades/quotes/nbbo (depending on the type)
    and return minute OHLC dataframe (this boils down to group by SQL query).

    :param file_name: TAQ daily file with all the trades
    :param tickers:
    :param year:
    :param type: ctm, cqm, nbbom
    :return:
    """
    try:
        print(file_name)

        if type == "ctm":
            query = """SELECT date,date_trunc('minute', time_m) AS minute,
                        sym_root,sum(size) AS agg_volume,
                        max(price) AS high, min(price) AS low,
                        (MAX(ARRAY[tr_seqnum, price]))[2] AS last,
                        (MIN(ARRAY[tr_seqnum, price]))[2] AS open
                        FROM taqm_{}.{}
                        WHERE sym_root in %(tickers)s 
                        GROUP BY minute, sym_root, date 
                        ORDER BY minute""".format(year, file_name)
        elif type == "nbbom":
            query = """SELECT date,date_trunc('minute', time_m) AS minute,
                        sym_root, max(best_bid) AS high_bid, min(best_bid) AS low_bid,
                        (MAX(ARRAY[qu_seqnum, best_bid]))[2] AS last_bid,
                        (MIN(ARRAY[qu_seqnum, best_bid]))[2] AS open_bid,
                        max(best_ask) AS high_ask, min(best_ask) AS low_ask,
                        (MAX(ARRAY[qu_seqnum, best_ask]))[2] AS last_ask,
                        (MIN(ARRAY[qu_seqnum, best_ask]))[2] AS open_ask
                        FROM taqm_{}.{}
                        WHERE sym_root in %(tickers)s 
                        GROUP BY minute, sym_root, date 
                        ORDER BY minute""".format(year, file_name)

        else:
            raise ValueError("Unknown type of query!")

        df_daily = db_cursor.raw_sql(query, params=tickers)
        df_daily.to_csv("{1}/{0}_{1}.csv".format(file_name, year))
        del df_daily
        return None
    except Exception as error:
        print("{} failed! Error {}".format(file_name, error))
    return None


def gather_dfs(dfs_):
    df_ohlc_ = pd.DataFrame(columns=["date", "minute", "sym_root", "agg_volume", "high", "low", "last", "open"])
    for df_ in dfs_:
        df_ohlc_ = df_ohlc_.append(df_, ignore_index=True)
    df_ohlc_.sort_values(by=["date", "minute", "sym_root"])
    return df_ohlc_


def set_global_cursor(db_):
    global db_cursor
    db_cursor = db_


if __name__ == '__main__':

    parallelize = True

    db = wrds.Connection()

    if parallelize:
        # pool = Pool(os.cpu_count())
        pool = Pool(initializer=set_global_cursor, initargs=(db,), processes=4)
        print("Number of cores: ", os.cpu_count())

    # could have outer for loop over years, but then the dataframe would be too large to store on the cloud (10GB limit)
    year_ = 2019
    assert year_def == year_

    # do we work with trades (ctm), quotes (cqm) or nbbo (nbbom) files
    type_ = "nbbom"
    assert type_def == type_

    files = [x for x in db.list_tables('taqm_{}'.format(year_)) if x.startswith(type_)]

    def process_daily_taq_wrapper(file_name):
        return process_daily_taq(file_name)

    print("start processing TAQ")
    if parallelize:
        dfs = pool.map(process_daily_taq, files)
    else:
        # set global cursor
        set_global_cursor(db)
        for file in files:
            print("Processing {}".format(file))
            start_time = timeit.timeit()
            process_daily_taq_wrapper(file)
            end_time = timeit.timeit()
            print("{} processed! Running time: {}.".format(file, end_time-start_time))

    db.close()
s