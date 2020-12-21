"""
This scipt is used to preprocess the textual data. The following steps are applied:
- Removing proper nouns
- Lowercasing
- Removing special characters, punctuations, symbols
- Removing stop-words
- Lemmatization
- Removing non-english words
"""

# General packages
from collections import Counter
from collections import defaultdict

# NLTK package
from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words as nltk_words
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import download as nltk_download
nltk_download('words')
nltk_download('averaged_perceptron_tagger')
nltk_download('wordnet')
nltk_download('stopwords')
nltk_download('punkt')

# Vectorizer
from sklearn.feature_extraction import DictVectorizer

lemmatizer = None
stop_words = None
tag_dict = None
english_words = None
def process_sentence(sent):
    token_list = word_tokenize(sent) # List: words
    token_list = pos_tag(token_list) # List: (word, tag)
    token_list = [(word.lower(), tag) for word, tag in token_list if tag != 'NNP' and tag != 'NNPS'] # Remove proper noun, lower case
    token_list = [(word, tag) for word, tag in token_list if word.isalpha()] # Delete special characters, symbols, punctuations
    token_list = [(word, tag) for word, tag in token_list if word not in stop_words] # Remove stop-words
    token_list = [lemmatizer.lemmatize(word, tag_dict[tag[0]]) for word, tag in token_list] # Lemmatization
    token_list = [word for word in token_list if word in english_words] # Keep only english words
    return Counter(token_list)

def process_text(text):
    sentence_list = sent_tokenize(text)
    sentence_list = map(process_sentence, sentence_list)
    token_counter = Counter()
    for counter in sentence_list:
        token_counter = token_counter + counter
    return token_counter

def process_df(df):
    # Input: chunk of the dataframe
    # Returns: list of Counters (word count), list of company counters, list of publication date values
    return df['text'].apply(process_text).values, df['company'], df['publish_date'], df['publish_date_utc'], df['url']

def initializer():
    # Initializer function for all the processes
    global tag_dict, lemmatizer, stop_words, english_words
    tag_dict = defaultdict(lambda: wordnet.NOUN)
    tag_dict["J"] = wordnet.ADJ
    tag_dict["V"] = wordnet.VERB
    tag_dict["R"] = wordnet.ADV
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    english_words = set(nltk_words.words())

if __name__ == "__main__":
    # Packages needed for running
    import multiprocessing as mp
    import pandas as pd
    import time
    from src.models.utils.large_pickle_io import write_large_pickle
    # Parameters
    data_path = '../../data/stage2_output.csv'
    num_rows = 10000 # If None then runs on all
    chunk_size = 500
    output_path = '../../data/word_count.p'

    # Read in the dataframe as an iterator to avoid loading it to memory
    df_iter = pd.read_csv(data_path,
                     nrows = num_rows,
                     index_col = 0,
                     chunksize = chunk_size)
    print('Data loaded')

    # Setup the multiprocessing pool
    pool = mp.Pool(initializer=initializer, initargs=())
    print('Number of cores in the system: ', mp.cpu_count())

    # Apply the process_df function on all the entries of the iterator
    start_time = time.time()
    results = pool.imap(process_df, df_iter)
    map_output = {'word_counts': [], 'company': [], 'publish_date': [], 'publish_date_utc': [], 'url': []}
    for out in results:
        map_output['word_counts'].extend(out[0])
        map_output['company'].extend(out[1])
        map_output['publish_date'].extend(out[2])
        map_output['publish_date_utc'].extend(out[3])
        map_output['url'].extend(out[4])
    print('\nTotal running time: {} seconds'.format(round(time.time() - start_time, 4)))

    # DictVectorizer
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(map_output['word_counts'])  # X: scipy.sparse.csr.csr_matrix
    print('Type of output: ', type(X))
    print('Shape of output: ', X.shape)
    print('Number of features: ', len(vectorizer.get_feature_names()))
    print('Short tokens: ', [word for word in vectorizer.get_feature_names() if len(word) <= 2])
    print('Length of company list: ', len(map_output['company']))
    print('Length of publish date list: ', len(map_output['publish_date']))

    output_data = {'count_matrix': X,
                   'columns': vectorizer.get_feature_names(),
                   'index': pd.DataFrame({'company': map_output['company'],
                                          'publish_date': map_output['publish_date'],
                                          'publish_date_utc': map_output['publish_date_utc'],
                                          'url': map_output['url']})}
    print(output_data['index'].head(10).to_string())

    # Save the output
    write_large_pickle(output_data, output_path)
