"""
This script contains a function that fits several models for the given training and validation data and returns the model fitting the best.
"""

import sys
# sys.path.append('src/models/benchmark')
# from sestm_model import SESTM_model
from src.models.sestm_model import SESTM_model
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from scipy.stats import kendalltau
from copy import deepcopy


def parameter_selection(X_train, y_train, X_valid, y_valid, optimizer="scipy",
                        logger=None, to_rank=False, verbose = False, refit=False, random_pred=True,
                        param_grid={'alpha': [25, 50, 100],  # 'alpha': [25, 50, 100],
                                    'kappa': [0.97, 0.98, 0.985],  # 'kappa': [0.96, 0.97, 0.98, 0.985],
                                    'lambda': [0.0, 0.25, 0.5, 0.75],  # 'lambda': [0.1, 0.25, 0.5],
                                    'lambda_reg': [0.0, 0.25, 0.75, 1.0]}
                        # 'lambda_reg': [0, 0.1, 0.5, 1]}
                        ):
    """

    :param X_train: scipy sparsematrix
    :param y_train: pd.Series
    :param X_valid: scipy sparsematrix
    :param y_valid: pd.Series
    :param optimizer: string in ['scipy', 'EM'] or integer
    :param logger: logger object
    :param verbose: True/False
    :param refit: whether or not to refit on entire data (train+validation) after the optimal hyperparam set was found
    :param random_pred: whether or not to calculate MAE also for random predictions (uniform on [0,1]).
        For model fit evaluation purposes.
    :return:
    """


    # param_grid = {'alpha': [50],
    #                 'kappa': [0.985],
    #                 'lambda': [0.5]
    # }

    # param_grid = {'alpha': [100],
    #               'kappa': [0.08, 0.06],
    #               'lambda': [1, 5, 10]
    # }


    best_model = None
    best_val_score = np.inf

    # Main iteration over the parameters
    for a in param_grid['alpha']:
        for k in param_grid['kappa']:
            for l2 in param_grid['lambda_reg']:
                # Fit model
                tmp_model = SESTM_model(alphaPlus = a,
                                        alphaMinus = a,
                                        kappa = k,
                                        penaltyLambda = 0.0,
                                        lambda_reg = l2,
                                        to_rank = to_rank,
                                        optimizer = optimizer)
                tmp_model.fit(X_train, y_train, verbose=False)
                for l in param_grid['lambda']:
                    if verbose:
                        if logger is None:
                            print(
                                'Fitting parameters alpha: {} kappa: {} lambda: {} lambda_reg: {}'.format(a, k,
                                                                                                          l,
                                                                                                          l2))
                        else:
                            logger.append(
                                'Fitting parameters alpha: {} kappa: {} lambda: {} lambda_reg: {}'.format(a, k,
                                                                                                          l,
                                                                                                          l2))
                    tmp_model.penaltyLambda = l # Set penaltyLambda to the right number
                    val_score = tmp_model.validation(X_valid, y_valid)
                    if verbose:
                        if logger is None:
                            print('Validation score: ', val_score)
                        else:
                            logger.append('Validation score: {}'.format(val_score))
                    if val_score < best_val_score:
                        best_model = deepcopy(tmp_model) # Deep copy so the penaltyLambda value does not get overwritten
                        best_val_score = deepcopy(val_score)

    # refit on train + validation set after optimal hyperparameters were found
    if refit:
        if logger:
            logger.append("Refitting the model with optimal hyperparams on train + valid data!")

        best_model.fit(vstack([X_train, X_valid]), pd.concat((y_train, y_valid), ignore_index=True))

    # random prediction - for evaluation of model fit
    if random_pred:
        y_valid_labels = best_model.estimate_p(y_valid[~y_valid.isna()], raw_ranks=to_rank)
        if to_rank:
            y_pred_random = np.random.permutation(y_valid_labels.shape[0]) + 1
            random_score = - kendalltau(y_valid_labels, y_pred_random)[0]
        else:
            y_pred_random = np.random.uniform(0, 1, y_valid_labels.shape)
            random_score = np.abs(y_valid_labels - y_pred_random).mean()
        if logger:
            logger.append("Best model score: {}. Random score: {}".format(best_val_score, random_score))
        else:
            print("Best model score: {}. Random score: {}".format(best_val_score, random_score))

    # TODO: compare distribution of labels and distribution of predictions

    return best_model, best_val_score


if __name__ == "__main__":
    from src.models.utils.large_pickle_io import *
    import gc
    import datetime
    import time

    # Load the data and drop rows without labels
    data_path = '../../../data/output2020/word_count_labeled.p'
    data = read_large_pickle(data_path)
    print('Number of observations: ', data['count_matrix'].shape[0])
    print('Number of words: ', data['count_matrix'].shape[1])

    # Drop NaN LABELS
    #idx_to_drop = ~data['index']['LABEL'].isna().values
    #print('Number of rows with NaN Label: ', sum(~idx_to_drop))
    #word_count = data['count_matrix'][idx_to_drop, :]
    #index_df = data['index'][idx_to_drop].reset_index(drop=True)
    word_count = data.pop('count_matrix')
    index_df = data.pop('index')
    columns = data.pop('columns')
    del data
    gc.collect()

    # Cutting the timeseries
    start_date = datetime.datetime.strptime('2016-09-01', '%Y-%m-%d').date()
    valid_date = datetime.datetime.strptime('2017-07-01', '%Y-%m-%d').date()
    test_date = datetime.datetime.strptime('2017-08-01', '%Y-%m-%d').date()

    # Trainind and validation indices
    train_idx = [i for i, date in enumerate(index_df['date_NYC']) if (date >= start_date and date < valid_date)]
    valid_idx = [i for i, date in enumerate(index_df['date_NYC']) if (date >= valid_date and date < test_date)]
    print('Number of training samples: ', len(train_idx))
    print('Number of validation samples: ', len(valid_idx))

    X_train = word_count[train_idx, :]
    y_train = index_df.loc[train_idx, 'LABEL']
    X_valid = word_count[valid_idx, :]
    y_valid = index_df.loc[valid_idx, 'LABEL']

    start_time = time.time()
    best_model, best_score = parameter_selection(X_train, y_train, X_valid, y_valid, verbose=True, optimizer= 200)
    print('Fitting time: ', round(time.time() - start_time, 2))
    print('Best score: ', best_score)
    print('Best alpha: ', best_model.alphaPlus)
    print('Best kappa: ', best_model.kappa)
    print('Best lambda: ', best_model.penaltyLambda)