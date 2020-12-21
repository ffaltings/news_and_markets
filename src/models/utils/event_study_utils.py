import datetime
import pandas as pd


def parse_time_fast(x):
    """
    Expected time format: HH:MM:SS
    """
    h, m, s = int(x[0:2]), int(x[3:5]), int(x[6:8])
    dt = datetime.time(hour=h, minute=m, second=s)
    return dt


def read_SPX_data(spx_2014_2019='../../../data/finance data/SPX minute/SPX_2004_2019.txt',
                  spx_2020='../../../data/finance data/SPX minute/SPX_2020_2020.txt'):
    """
    Read in minute level spx data

    :param spx_2014_2019:
    :param spx_2020:
    :return:
    """

    # import SPX data and parse dates
    data = pd.read_csv(spx_2014_2019, sep=",", parse_dates=[0], header=None,
                       index_col=0)
    data2 = pd.read_csv(spx_2020, sep=",", parse_dates=[0], header=None,
                        index_col=0)

    # we only work with close prices on minute level
    data.drop([1, 2, 3], axis=1, inplace=True)
    data.columns = ["close"]

    data2.drop([1, 2, 3], axis=1, inplace=True)
    data2.columns = ["close"]

    return data.append(data2)


def binning_times(bin_size_minutes):
    market_open = datetime.time(9, 30, 0)
    market_close = datetime.time(16, 0, 0)
    times = []
    time = (datetime.datetime.combine(datetime.date(1, 1, 1), market_open)).time()
    while time <= (datetime.datetime.combine(datetime.date(1, 1, 1), market_close)).time():
        times.append(time)
        time = (datetime.datetime.combine(datetime.date(1, 1, 1), time) + datetime.timedelta(
            minutes=bin_size_minutes)).time()
    return times


def binning(x, times):
    for i in range(1, len(times)):
        if times[i - 1] <= x <= times[i]:
            return int(i - 1)