import pandas as pd
import numpy as np
import datetime
from dateutil.rrule import rrule, MONTHLY
import itertools
import matplotlib.pyplot as plt

from src.models.sestm_model import *
from src.models.utils.large_pickle_io import *
from src.models.sestm_model import *
from src.models.model_param_search import *


def fit_and_predict(X_train, y_train, X_pred, y_pred):
    """
    Given the training data it outputs the predictions for X_pred for all possible combinations in param_grid
    :param X_train:
    :param y_train:
    :param X_pred:
    :param y_pred:
    :return:
    """
    # Iterate over the training parameters
    output_df = pd.DataFrame(np.nan,
                             index=y_pred.index,
                             columns=["alpha: {}, kappa: {}, lambda_pred: {}, lambda_fit: {}, rank: {}".format(a,b,c,d,e)
                                             for a,b,c,d,e in itertools.product(*param_grid.values())])

    for alpha, kappa, lambda_fit in itertools.product(param_grid['alpha'], param_grid['kappa'], param_grid['lambda_fit']):
        # Fit the model with the given values
        model = SESTM_model(alphaPlus = alpha,
                            alphaMinus = alpha,
                            kappa = kappa,
                            penaltyLambda = 0.0,
                            lambda_reg = lambda_fit,
                            to_rank = False,
                            optimizer = 500)
        model.fit(X_train, y_train)

        # Plot the number of sentiment charged words
        charged_words = X_pred[:,model.S_hat].sum(axis=1)[:,0]
        print(charged_words.shape)
        print('Parameters alpha: {}, kappa: {}, lambda_fit: {}'.format(alpha, kappa, lambda_fit))
        print('Mean: {}, Median: {}'.format(np.mean(charged_words), np.median(charged_words, axis=0)))
        plt.hist(charged_words, bins = 40, range=(0,200))
        plt.title('Parameters alpha: {}, kappa: {}, lambda_fit: {}'.format(alpha, kappa, lambda_fit))
        plt.show()

"""
        # Iterate over the prediction parameters
        for lambda_pred, to_rank in itertools.product(param_grid['lambda_pred'], param_grid['to_rank']):
            model.penaltyLamda = lambda_pred
            model.to_rank = to_rank
            pred = model.predict(X_pred)
            # Add to the output dataframe
            output_df.loc[:, "alpha: {}, kappa: {}, lambda_pred: {}, lambda_fit: {}, rank: {}".format(alpha,
                                                                                                      kappa,
                                                                                                      lambda_pred,
                                                                                                      lambda_fit,
                                                                                                      to_rank)] = pred

    return output_df
"""

if __name__ == "__main__":
    # paths
    word_count_path = '../../data/word_count_labeled.p'

    # Timespan parameters
    START_DATE = datetime.strptime("2016-09-01", "%Y-%m-%d").date()
    END_DATE = datetime.strptime("2020-02-27", "%Y-%m-%d").date()
    training_months = 10
    validation_months = 5
    prediction_months = 1
    expanding = False

    # Parameter grid
    param_grid = {'alpha': [25, 50, 100],
                  'kappa': [0.96, 0.97, 0.98, 0.985],
                  'lambda_pred': [0.1, 0.25, 0.5],
                  'lambda_fit': [0],
                  'to_rank': [True, False]}
                  # 'lambda_reg': [0, 0.1, 0.5, 1]}

    # Reading in the data
    data = read_large_pickle(word_count_path)
    index_df = data.pop('index')
    column_names = data.pop('columns')
    count_matrix = data.pop('count_matrix')
    print('Count matrix shape: ', count_matrix.shape)
    print('Column names length: ', len(column_names))
    print('Index df shape: ', index_df.shape)
    print(index_df.head())

    # Initializing first days of the months
    first_days_list = [dt.date() for dt in rrule(MONTHLY, dtstart=START_DATE, until=END_DATE)]
    print('First days: ', first_days_list[:5])

    # Initialize a dataframe for all the predictions
    predictions_df = pd.DataFrame(np.nan,
                                  index = index_df.index,
                                  columns = ["alpha: {}, kappa: {}, lambda_pred: {}, lambda_fit: {}, rank: {}".format(a,b,c,d,e)
                                             for a,b,c,d,e in itertools.product(*param_grid.values())])

    # Training and validation masks
    for i, date in enumerate(first_days_list[:-(training_months + validation_months + prediction_months)]):
        print('----- TIMESTEP: {} -----'.format(date))
        print('Training period end: ', str(first_days_list[i+training_months]))
        print('Validation period end: ', str(first_days_list[i+training_months+validation_months]))

        # Subset the data for training, validation and prediction
        if expanding:
            training_idx = np.where(index_df['date_NYC'].apply(lambda x: x < first_days_list[i + training_months]))[0]
        else:
            training_idx = np.where(index_df['date_NYC'].apply(lambda x: first_days_list[i] <= x
                                                                         and x < first_days_list[i + training_months]))[0]
        validation_idx = np.where(
            index_df['date_NYC'].apply(lambda x: first_days_list[i + training_months] <= x
                                                 and x < first_days_list[i + training_months + validation_months]))[0]
        prediction_idx = np.where(
            index_df['date_NYC'].apply(lambda x: first_days_list[i + training_months + validation_months] <= x
                                                 and x < first_days_list[i + training_months + validation_months + prediction_months]))[0]


        X_train = count_matrix[training_idx, :]
        y_train = index_df.loc[training_idx, 'LABEL']
        X_valid = count_matrix[validation_idx, :]
        y_valid = index_df.loc[validation_idx, 'LABEL']
        #X_pred = count_matrix[prediction_idx, :]

        # Predict for all combinations
        output = fit_and_predict(X_train, y_train, X_valid, y_valid)
        print(output)
        quit()
