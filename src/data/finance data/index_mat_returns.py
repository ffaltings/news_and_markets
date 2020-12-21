"""
Adds the returns to index matrix (later used in absorption analysis - linear regression).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def open_ohlc_data(filepath):
	df = pd.read_csv(filepath)
	df['minute'] = df['minute'].apply(lambda x: x.split()[-1].split('.')[0])
	df = df.sort_values(by=['sym_root', 'minute']).set_index(['sym_root'], drop=True).drop('Unnamed: 0', axis=1)
	return df


def filepath_from_date(date, data_dir):
	return os.path.join(data_dir, *datetime.datetime.strftime(date, '%Y ctm_%Y%m%d_%Y.csv').split())


def get_open_price(df, ticker, time):
	try:
		idx = df.loc[ticker]['minute'].searchsorted(str(time), side='right') - 1
	except KeyError:
		return -1

	if idx < 0: return -1

	today = datetime.date(1, 1, 1)
	found_time = datetime.datetime.strptime(df.loc[ticker]['minute'].iloc[idx], '%H:%M:%S').time()
	delta = datetime.datetime.combine(today, time) - datetime.datetime.combine(today, found_time)
	if delta > TIME_LOOKBACK: return -1

	return df.loc[ticker].iloc[idx]['open']


def partial_fill_table(table, date, deltas, day_deltas, idx, df, start_time, end_time, time_lookback, opens, close, verbose=0):
	counter  = 0
	for row in idx.loc[date].iterrows():
		ticker = row[1]['TICKER']
		time = row[1]['time_NYC']
		i = row[1]['id']

		if time < start_time or time > end_time: continue

		today = datetime.date(1, 1, 1)  # only considering articles on same day so just use dummy date
		time_plus_deltas = [(datetime.datetime.combine(today, time) + d).time() for d in deltas]

		price0 = get_open_price(df, ticker, time)
		pdeltas = [get_open_price(df, ticker, t) for t in time_plus_deltas]
		# add +1 day open and close
		for delta in day_deltas:
			date_plus_delta = date + delta
			try:
				pdeltas.append(opens.loc[datetime.datetime.strftime(date_plus_delta, '%Y-%m-%d')][ticker])
			except KeyError:
				pdeltas.append(-1)
			try:
				pdeltas.append(close.loc[datetime.datetime.strftime(date_plus_delta, '%Y-%m-%d')][ticker])
			except KeyError:
				pdeltas.append(-1)

		if verbose > 1: print(
			"row: {}, ticker: {}, time: {}, price: {}, price+15: {}".format(i, ticker, time, price0, pdeltas[0]))

		table[i, 0] = price0
		for j, p in enumerate(pdeltas):
			table[i, 1 + j] = p

		counter += 1
	print('Found: {} returns'.format(counter))

if __name__ == '__main__':
	TRADE_START = datetime.time(9, 20, 0)
	TRADE_END = datetime.time(13, 0, 0)
	TIME_LOOKBACK = datetime.timedelta(0, 30 * 60)  # 30 min lookback

	DELTAS = [datetime.timedelta(0, 15 * 60), datetime.timedelta(0, 30 * 60), datetime.timedelta(0, 60 * 60), datetime.timedelta(0, 3*60*60)]
	DAY_DELTAS = [datetime.timedelta(1,0), datetime.timedelta(2,0), datetime.timedelta(5,0), datetime.timedelta(15,0)]
	TABLE_COLUMNS = ['+0', '+15min', '+30min', '+1h', '+3h', '+1day_open', '+1day_close',
					 '+2day_open', '+2day_close',  '+5day_open', '+5day_close',  '+15day_open', '+15day_close']
	TABLE_COLUMNS = ['price' + s for s in TABLE_COLUMNS]

	data_dir = 'data/external/ohlc'
	index_mat_path = 'data/index_matrices/index_matrices_with_sentiment/EW_LS_2019-12-08_18-06-24_index_matrix.p'
	index_save_path = 'data/index_matrices/index_matrices_with_sentiment/EW_LS_2019-12-08_18-06-24_index_matrix_with_returns.p'

	idx = np.load(index_mat_path, allow_pickle=True)
	idx['time_NYC'] = idx['time_NYC'].apply(lambda x: x.replace(second=0, microsecond=0))
	idx = idx.sort_values(by=['date_NYC', 'time_NYC'], ascending=False).set_index(['date_NYC'])
	idx = pd.concat([idx, pd.DataFrame(np.arange(idx.shape[0]), columns=['id'], index=idx.index)], axis=1)
	dates = np.sort(np.unique(idx.index.get_level_values('date_NYC')))[::-1]

	open_prices = pd.read_csv('data/external/yahoo_finance/open_prices.csv', index_col='date').drop('Unnamed: 0',
																									   axis=1)
	close_prices = pd.read_csv('data/external/yahoo_finance/adj_close_prices.csv', index_col='date').drop(
		'Unnamed: 0', axis=1)

	table = -1 * np.ones((idx.shape[0], len(DELTAS) + 1 + 2*len(DAY_DELTAS)))

	for date in dates:
		filepath = filepath_from_date(date, data_dir)
		print(filepath)
		try:
			df = open_ohlc_data(filepath)
		except FileNotFoundError:
			print('FileNotFound')
			continue

		partial_fill_table(table, date, DELTAS, DAY_DELTAS, idx, df, TRADE_START, TRADE_END, TIME_LOOKBACK, open_prices, close_prices)

	table = pd.DataFrame(table, columns=TABLE_COLUMNS, index=idx.index)
	new_idx = pd.concat([idx, table], axis=1)
	new_idx.to_pickle(index_save_path)

