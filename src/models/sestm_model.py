from scipy.optimize import minimize
from scipy.stats import kendalltau
import multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import warnings
import time
import random


class SESTM_model(object):
    def __init__(self, alphaPlus, alphaMinus, kappa, penaltyLambda, lambda_reg, to_rank, optimizer="scipy"):
        """
        Input parameters.
        Types can be:
            - alphaPlus, alphaMinus: float (same as in the paper) and kappa: integer (threshold of frequency)
            - alphaPlus, alphaMinus: integer (size of the final set) and kappa: float (fraction of words included)
        :param alphaPlus: float or integer
        :param alphaMinus: float or integer
        :param kappa: float or integer
        :param penaltyLambda: float
        :param lambda_reg: regularization hyperparam when fitting O matrix
        :param optimizer: which optimizer to use in model's predict function,
                            scipy or int (size of the grid in grid search)
        """
        self.alphaPlus = alphaPlus
        self.alphaMinus = alphaMinus
        self.kappa = kappa
        self.penaltyLambda = penaltyLambda
        self.lambda_reg = lambda_reg
        self.to_rank = to_rank
        self.S_hat = None
        self.O_hat = None
        assert optimizer in ["scipy", "EM"] or type(optimizer) is int
        self.optimizer = optimizer

    def screening(self, X, y, target_period = None, index = None):
        """
        Function for STEP 1
        :param X: scipy.sparse matrix with shape (n,m) articles x words
        :param y: np.array with corresponding training returns
        :param target_period: Length of prediction period eg. "M" for one month of "3M" for 3 months
        :param index: index array of X
        :return: A set of indices corresponding to the sentiment charged words
        """
        # Input shapes
        n_row, n_col = X.shape

        # Calculate k (article counts per word)
        X_nonzero = X.nonzero() # Length 2 tuple with nonzero indices: (array of rows, array of columns)
        wordIndex, k = np.unique(X_nonzero[1], return_counts = True) # wordIndex is an array of column indices, k is the occurance of that column

        # Calculate the signs of y
        positive_rows = np.where(y > 0)[0] # Shape: (n_row,), indices of rows with positive label

        # Subset wordIndex and k (positive article counts per word)
        row_mask = np.isin(X_nonzero[0], positive_rows) # rows that we consider to calculate f_j
        wordIndex_pos, k_pos = np.unique(X_nonzero[1][row_mask], return_counts = True)

        # Calculate f (Proportion of positive sign articles out of all articles mentioning a word)
        f = np.zeros(n_col) # Shape: (n_col,), float
        f[wordIndex_pos] = k_pos # Add the denominator, float
        f[wordIndex] = f[wordIndex] / k # Divide by k, float

        # ----- Calculate S_hat -----
        # Calculate the frequent sent (store as an np.array)
        if type(self.kappa) == int:
            freq_set = wordIndex[k >= self.kappa]  # set of words frequent enough
        elif type(self.kappa) == float:
            assert 0 <= self.kappa <= 1
            # If target period and index is provided always keep the words that are in the top self.kappa percentile each period
            if (target_period is not None) and (index is not None):
                # Find the indices of each period
                resampleObject = pd.Series(1, index=pd.DatetimeIndex(index)).resample(target_period)
                resampleIndexDict = resampleObject.indices # Dictionary with {period: [indices in that period]}
                freq_set = set()
                # For each period find the upper self.kappa*100 percentile of the words and add them to freq_set
                for period in resampleIndexDict:
                    row_mask = np.isin(X_nonzero[0], resampleIndexDict[period])  # rows that we consider to calculate f_j
                    words_period, k_period = np.unique(X_nonzero[1][row_mask], return_counts=True)
                    k_period_full = np.zeros(n_col)
                    k_period_full[words_period] = k_period
                    kappa_period = np.percentile(k_period_full, self.kappa * 100)
                    freq_set = freq_set.union(set(words_period[k_period >= kappa_period]))  # Selected words of the period
                freq_set = np.array(list(freq_set))
            else:
                # Keep the top self.kappa percentage of the words by frequency in the whole dataset
                # Sort the words in ascending order by frequency and drop the non-frequent ones
                freq = np.zeros(n_col)
                freq[wordIndex] = k
                kappa = np.percentile(freq, self.kappa * 100)
                freq_set = wordIndex[k >= kappa] # np.array with the frequent indicies

        # Calculate the positive and the negative sets
        if type(self.alphaPlus) == float and type(self.alphaMinus) == float:
            # Positive set of words
            self.positive_set = np.where(f >= 0.5 + self.alphaPlus)[0]
            self.positive_set = self.positive_set[np.isin(self.positive_set, freq_set)]
            self.positive_f = f[self.positive_set]
            # Negative set of words
            self.negative_set = np.where(f < 0.5 + self.alphaMinus)[0]
            self.negative_set = self.negative_set[np.isin(self.negative_set, freq_set)]
            self.negative_f = f[self.negative_set]
            # Concatenate
            self.S_hat = np.concatenate([self.negative_set, self.positive_set])
        elif type(self.alphaPlus) == int and type(self.alphaMinus) == int:
            # Take the most positvie/negative out of the frequent set
            f_argsort = np.argsort(f) # Return the arguments in increasing order
            f_argsort = f_argsort[np.isin(f_argsort, freq_set)] # Remove entries not frequent enough
            self.positive_set = f_argsort[-self.alphaPlus:]
            self.positive_f = f[self.positive_set]
            self.negative_set = f_argsort[:self.alphaMinus]
            self.negative_f = f[self.negative_set]
            self.S_hat = np.concatenate([self.negative_set, self.positive_set])
        else:
            raise TypeError

    def estimate_p(self, arr, raw_ranks=False):
        """
        Helper function to estimate the p value for the financial labels
        p_hat = rank(y)/n
        :param arr: labels, pd.Series
        :param raw_ranks: if true return raw (unnormalized) ranks, else return normalized ranks
        :return: normalized ranks, np.Array
        """
        temp = arr.argsort() # Returns -1 for NaN values
        assert sum(temp == -1) == 0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(arr)) + 1
        if raw_ranks:
            return ranks
        else:
            return ranks / arr.shape[0]  # Shape: (n,)

    def fittingSentiments(self, X, y):
        """
        Function for STEP 2
        :param X: scipy sparse matrix with count vectorizer rows, Shape: (n,m)
        :param y: np.array with corresponding training returns, Shape: (n,)
        :return: np.array of shape (n,2) (positive sentiments, negative sentiments)
        """
        # Estimate p vector
        p = self.estimate_p(y) # Shape: (n,)

        # Sum of counts in sentiment charged words
        s_hat = X[:, self.S_hat].sum(axis = 1) # Shape: (n, 1)
        s_hat[s_hat == 0] = 1 # If s_hat is zero it means that the whole row is zero, to avoid division by zero change 0 to 1

        # Estimate D_hat matrix
        D_hat = (X[:, self.S_hat]/s_hat).transpose() # Shape: (|\hat{S}|, n)

        # Estimated W_hat
        W_hat = np.vstack([p, 1-p]) # Shape: (2,n)
        W_inverse = np.linalg.inv(W_hat.dot(np.transpose(W_hat)) - self.lambda_reg*X.shape[0]*np.array([[1,-1],[-1,1]])) # Shape: (2,2)

        # Estimate the parameters
        self.O_hat = D_hat.dot(W_hat.transpose()).dot(W_inverse) # Shape: (n,2)

        # Set negative values to zero and normalize
        self.O_hat[self.O_hat < 0] = 0
        self.O_hat = self.O_hat / self.O_hat.sum(axis = 0)

    def fit(self, X, y, verbose = True, target_period = None, X_index = None):
        assert X.shape[0] == y.shape[0]
        mask = np.where(~y.isna())[0]
        if verbose:
            print('Number of NaN label: ', y.shape[0] - len(mask))
        # Sklearn style fitting function that combines the two steps
        if verbose:
            print('Screening the sentiment charged words')
        self.screening(X[mask, :], y.iloc[mask], target_period = target_period, index = X_index)
        if verbose:
            print('Fitting Sentiments')
        self.fittingSentiments(X[mask], y.iloc[mask])

    def optimize_thread(self, idx, X):
        """
        function to feed to multiprocessing pool, MLE approach
        :param idx: index of the target input
        :param X: Input word count matrix
        :return: p
        """
        # Row and columns of interest
        slice = X[idx, self.S_hat]  # Shape: (1, |S_hat|)
        total_count = slice.sum()
        def f(p):
            p = p[0] # Only subtract from the input array
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sentiment = np.log(p*self.O_hat[:,0] + (1-p)*self.O_hat[:,1]) # Shape: (|S_hat|, 1)
                summation = slice.dot(sentiment)[0,0] # Float
                return -(summation / total_count + self.penaltyLambda * np.log(p * (1 - p))) # Float

        # Optimization
        res = minimize(f, x0 = np.array([0.5, ]), method='L-BFGS-B', bounds=((0, 1),))

        if res.success:
            return res.x[0]
        else:
            return None

    def optimize_thread_grid(self, X):
        """
        Function that finds the global minimum by grid search, MLE approach
        :param X: Input word count matrix with only sentiment charged columns
        :param r: granuality of the linear space between 0 and 1
        :return: np.array with the optimal p values
        """
        r = self.optimizer

        # Row and columns of interest
        total_count = X[:, self.S_hat].sum(axis=1) # Shape: (n, 1)

        # p linspace
        p = np.linspace(0.001, 1, r, endpoint=False).reshape(1, -1)  # Shape: (1, r)

        with warnings.catch_warnings():
            sentiment = np.log(self.O_hat[:, 0].dot(p) + self.O_hat[:, 1].dot(1 - p)) # Shape: (|S_hat|, r)
            summation = X[:, self.S_hat].dot(sentiment) / total_count + self.penaltyLambda * np.log(p * (1 - p)) # Shape: (n, r)
            argmax = np.argmax(summation, axis=1) # Shape: (n,1)
            p_hat = p.reshape(-1)[argmax] # Shape: (n,1)
            return p_hat.reshape(-1)

    def optmizer_em(self, X):
        """
        Function that uses the expectation maximization formula
        :param X: Input word count matrix with only sentiment charged columns, Shape: (n,m)
        :return: numpy array, Shape: (n,)
        """
        tone = (self.O_hat[:,0] - self.O_hat[:,1]).A.reshape(-1) # Shape: (|S_hat|, )
        p_hat = (self.penaltyLambda/2 - tone.T.dot(self.O_hat.A[:,1]))*np.ones(X.shape[0]) # Shape: (n,)
        p_hat = p_hat + X[:, self.S_hat].dot(tone).reshape(-1) # Shape: (n,)
        p_hat = p_hat / (self.penaltyLambda + np.sum(tone ** 2))
        return p_hat.reshape(-1)

    def predict(self, X_new, raw_ranks=False):
        """
        Predict function
        :param X_new: scipy sparse matrix with dimensions either 1 or 2
                (if 2 then it is similar to the fitting matrix i.e articles x word counts)
        :param raw_ranks: if true return raw (unnormalized) ranks, else return normalized ranks
        :return: np.array with the predicted sentiment
        """

        solver = self.optimizer

        if len(X_new.shape) == 1:
            X_new = X_new.reshape(1, -1)  # Shape: (1,m)

        # Initialize predictions as 0.5
        predictions = np.full(X_new.shape[0], 0.5)
        non_zero_idx = np.where(X_new[:, self.S_hat].sum(axis=1) > 0)[0]

        # Apply the different function to solve
        if solver == 'scipy':
            # Multiprocessing optimization
            pool = mp.Pool()
            p_list = pool.starmap(self.optimize_thread, [(i, X_new) for i in non_zero_idx])
            pool.close()
        elif isinstance(solver, int):
            p_list = self.optimize_thread_grid(X_new[non_zero_idx, :])
        elif solver == 'EM':
            p_list = self.optmizer_em(X_new[non_zero_idx, :])

        if self.to_rank:
            p_list = self.estimate_p(p_list, raw_ranks=raw_ranks)

        # Add the values for the non-zero rows
        predictions[non_zero_idx] = p_list
        return predictions

    def validation(self, X, y, verbose = False):
        """
        Evaluating the current model on X and y
        :param X: scipy sparse matrix with selected rows and all the columns
        :param y: training labels for the selected rows, pd.Series
        :return: l1 loss of the model
        """
        # Check for nan labels
        assert X.shape[0] == y.shape[0]
        mask = np.where(~y.isna())[0]
        if verbose:
            print('Number of NaN label: ', y.shape[0] - len(mask))

        # Target values
        p = self.estimate_p(y.values[mask], raw_ranks=self.to_rank)
        pred = self.predict(X[mask, :], raw_ranks=self.to_rank)
        mask2 = (pred != np.array(None))  # Remove non-converged predictions (only for scipy optimizer)

        if self.to_rank:  # comparing lists of raw ranks, Kendall tau distance
            err = - kendalltau(p[mask2], pred[mask2])[0]
        else:  # L1 loss
            err = np.abs(p[mask2] - pred[mask2]).mean()
        return err

    def get_trading_weights(self, today, last, weights_method, strategy, nr_companies,
                            columns, index_matrix, news_matrix, logger, recent_news_only):
        """
        Calculate trading weights based on the current news. Used in back-testing (see backtesting.py).

        :param today: day of trading
        :param last: previos trading day (for Monday, last trading day is Friday)
        :param weights_method: equal-weights (EW) or value-weights (VW)
        :param strategy: long-short (LS), long (L), short (S)
        :param nr_companies: number of companies to invest in
        :param columns: columns of dataframe with open-to-open returns, contains all tickers
        :param index_matrix: index matrix (LABEL, TICKER, date_NYC, time_NYC, datetime_NYC)
        :param news_matrix: word count matrix
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
        index_matrix = index_matrix[(index_matrix["datetime_NYC"] <= today_dt) &
                                    (index_matrix["LABEL"].notnull()) & (index_matrix["datetime_NYC"] >= last_dt)]
        selected_tickers = list(index_matrix.TICKER)
        ind = list(index_matrix.index)

        if len(ind) > 0:

            # predict sentiment for news for this trading period
            selected_news = news_matrix[ind, :]
            start_time = time.time()
            pred_vector = self.predict(selected_news)
            end_time = time.time()
            # pred_vector = np.random.uniform(size=len(ind))
            pred_ticker_vector = list(zip(selected_tickers, pred_vector))

            logger.append("Nr. of news for period from {} to {}: {}".format(last_dt, today_dt, len(pred_ticker_vector)))

            # removing news for which predict returned None
            pred_ticker_vector = list(filter(lambda x: x[1] is not None, pred_ticker_vector))
            # logger.append("Predicted sentiment per news: {}".format(str(pred_ticker_vector)))
            # logger.append("Nr. of news on {} with prediction {}: ".format(today, len(pred_ticker_vector)))

            # def sentiment_sharpe(x):
            #     if np.std(x) == 0:
            #         return np.mean(x)
            #     else:
            #         return np.mean(x)/np.std(x)

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
                if 2*nr_companies > len(argsort_sentiment):
                    nr_companies = int(len(argsort_sentiment)/2)
                long = argsort_sentiment[-nr_companies:]
                short = argsort_sentiment[:nr_companies]

            long_companies = [tickers[i] for i in long]
            short_companies = [tickers[i] for i in short]
            # logger.append("Long companies : {}".format(str(long_companies)))
            # logger.append("Short companies : {}".format(str(short_companies)))

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
            ind = None
            pred_vector = None

        return weights_today, ind, pred_vector
