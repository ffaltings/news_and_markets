"""
This script is for the second stage of the general preprocessing pipeline
"""

import os
from os.path import join
import time
import datetime
from dateutil.tz import tzutc

import gzip
import pandas as pd
import multiprocessing as mp

from src.data.utils.pipeline import preprocess_pipeline
from src.data.utils.transformation import *
from src.data.utils.filter import *

pipeline = None
def process_file(path):
    """
    Processing one file
    :param path: path to the file
    :return: list of dictionaries, statistics dictionary
    """
    # File Statistics
    statistics = {}
    dropped = 0
    all_counter = 0
    kept_counter = 0
    start_time = time.time()
    pipeline.reinitialize_counter()  # Reinitialize the counter

    # Iterate over the lines
    output_list = []
    with gzip.GzipFile(path, 'r') as f:
        json_str = f.read().decode('utf-8')
        for line in json_str.splitlines():
            all_counter += 1
            try:
                eval_line = eval(line) # All elements of the dict is unstringified
                if pipeline(eval_line) is not None:
                    eval_line['crawl_time'] = path.split('-')[-2]
                    output_list.append(eval_line)
                    kept_counter += 1
            except Exception as e:
                print(e, line)
                dropped += 1

    # Add statistics
    statistics['run_time'] = time.time() - start_time
    statistics['num_records'] = all_counter
    statistics['num_kept'] = kept_counter
    statistics['syntax_drop'] = dropped
    statistics['file'] = path
    counter = pipeline.get_counter()
    for idx in counter:
        statistics[idx] = counter[idx]

    return output_list, statistics

def initializer(input_list):
    """
    Initializer for the processes
    :param input_list: list of filter/transformation objects as an input for the pipeline class
    :return: None
    """
    global pipeline
    pipeline = preprocess_pipeline(input_list)

if __name__ == "__main__":
    # Boolean whether to use multiprocessing
    multiprocess = True
    input_dir = "../../data/CCNewsProcessed"
    output_path = '../../data/stage2_statistics.csv'
    output_path_statistics = '../../data/stage2_output.csv'
    finance_model_path = '../../models/pipeline_v1.2.joblib'

    # Get a list of the paths to the files
    json_path_list = list()
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        json_path_list += [join(dirpath, file) for file in filenames]
    print('Number of input files: ', len(json_path_list))
    print('First 5 Json paths: ', json_path_list[:5])

    # Initializing the pipeline for the second stage
    input_list = [
        company_drop(['Facebook, Inc.', 'Twitter', 'Fox']),
        keyword_threshold_filter(),
        date_filter(utc_check='utc_full'),
        finance_filter(finance_model_path, cutoff_value = 0.35)
    ]

    # MULTIPROCESSING
    if multiprocess:
        print('Number of cores in the system: ', mp.cpu_count())
        # Multiprocessing pool
        pool = mp.Pool(initializer=initializer, initargs=(input_list,))

        # Use pool map
        start_time = time.time()
        record_list, statistics_list = zip(*pool.map(process_file, json_path_list))
        print('\n Total running time: {} seconds'.format(round(time.time() - start_time, 4)))
        pool.close()

        # Flatten the record_list
        record_list = [item for sublist in record_list for item in sublist]

    else:
        pipeline = preprocess_pipeline(input_list)
        print('Pipeline initialized')

        # Main loop over the list of paths
        start_time = time.time()
        record_list = []
        statistics_list = []
        for path in json_path_list[:10]:
            # Process the file
            file_list, stat_dict = process_file(path)
            record_list += file_list
            statistics_list.append(stat_dict)
        print('\n Total running time: {} seconds'.format(round(time.time() - start_time, 4)))

    # Statistics dictionary
    stat_df = pd.DataFrame(statistics_list)
    print(stat_df.head().to_string())
    print(stat_df.describe())
    stat_df.to_csv(output_path_statistics)

    # Convert dict_list to a Pandas DataFrame
    df = pd.DataFrame(record_list)
    print('Columns of the dataframe: ', df.columns)
    print('Shape of the dataframe: ', df.shape)
    print(df.head().to_string())
    df.to_csv(output_path)