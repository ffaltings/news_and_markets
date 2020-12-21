# On the impact of publicly available news and information transfer to financial markets

Code to accompany: 

## Scripts to reproduce results

### CommonCrawl Processing
Sections follow the workflow shown on Figure 1.

First fill out the lists in `data/sp500_list_25-02-2020.txt` corresponding to the `keywords` attribute
with the keywords corresponding to the respective companies.
#### Pre-Screening

Our pre-screening code is a Spark application that processes raw WARC files from common crawl,
performs initial filtering, and outputs news articles in jsonl files.
Because Common Crawl is hosted on Amazon S3, our code was run on AWS EMR, and reads and writes 
to and from S3. It can also be adapted to run on other infrastructure. Our code was cloned and
adapted from https://github.com/commoncrawl/cc-pyspark. 

Set the paths in lines 384-286 in `src/data/ccspar.py`. You need to point the script
to an s3 path for a `.txt` file containing the s3 paths of the Common Crawl WARC files to process. 
See https://commoncrawl.org/the-data/get-started/ to retrieve WARC paths. You also need to
set an output directory (this logs the s3 requests made by the script), and an s3 bucket
where the script will write the processed data to. By default it will write to 
`your_s3_bucket/CCNewsProcessed/run_timestamp`. Finally, you need to point the script to
the `sp500_list_25-02-2020.txt` file from step 1, and indicate the number of input WARC files to process.

The Spark application can be run with
```bash
python ccspark.py
```

#### Filtering
Set the `input_dir`, `output_statistics_path` and `output_file_path` variables at line 76 - 79 in the
`src/data/filtering_stage.py` file. Then execute
```bash
python filtering_state.py
```
from the `src/data` directory. The `input_dir` path should be the output of the Pre-Screeining step
i.e. `your_s3_bucket/CCNewsProcessed`.

**Remark**: This script has been implemented to be able to run on several CPU cores.
If you have sufficient local computational power, it can be run locally with the data downloaded from S3.
It is recommended to have at least 8 cores.

#### Alignment
Set the variables `data_path` and `output_path` at line 232-233 in the `src/data/news_alignment.py`. Then execute
```bash
python news_alignment.py
```
from the `src/data` directory. The `data_path` path should be the `output_file_path` from the Filtering step.

#### Pre-Processing
Set the variables `data_path` and `output_path` at line 79-80 in the `src/models/benchmark/utils/text_preprocessing.py`. Then execute
```bash
python text_preprocessing.py
```
from the `src/models/benchmark/utils` directory. The `data_path` should be the `output_path` from the Alignment step.
The output of this file is python dictionary as a pickle file with the following attributes
* `count_matrix`: numpy matrix with word counts
* `columns`: list of words corresponding to the columns in `count_matrix`
* `index`: pandas DataFrame with values: `company`, `publish_date`, `publish_date_utc`, `url`.
Each row of this DataFrame corresponds to a row in `count_matrix` in the same order.

### Finance Data Processing

To download financial data run
```bash
python finance_utils_yf.py
```
from `src/data/finance data`.

To add financial labels to the index matrix (see Pre-Processing step above) run
```bash
python finance_preprocess.py
```
from `src/models/utils/`. Set the paths for input datasets in lines 154-158.

### Transfer Entropy
Re-create conda environment 
```bash
conda create --name myenv --file /src/models/transfer_entropy/conda_transfer_entropy_analysis_environment.yml
```
Set the paths:

PATH_NBBO_RETURNS = "data/finance_data/minute_OHLC/NBBO/NBBO_parsed_v2.csv"

PATH_SENTIMENT_INDEX = "data/index_matrices_2020/EW_LS_2020-06-17_18-09-14_index_matrix.p"

in the script /src/models/transfer_entropy/transfer_entropy_intraday-2018-2020-polished.ipynb

The input files should be intra-day sentiment extracted data and intra-day financial returns.

### Backtesting

To run a backtest execute
```bash
python backtesting.py
```
from `src/models/`. Set the paths for input datasets in lines 736-738. 
Backtest configuration can be set with parameters in lines 759-779. 
For description of parameters see lines 342-375.
