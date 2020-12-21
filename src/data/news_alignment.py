import pandas as pd
import time
from collections import Counter
import spacy
import multiprocessing as mp
from nltk import sent_tokenize
from fuzzywuzzy import fuzz
import numpy as np
from scipy.sparse.csgraph import connected_components

from src.data.utils.keywordsIO import *

# Variables for multiprocessing
keyword_dict = None
nlp = None
counter_threshold = 5
publisher_list = ["Facebook, Inc.", "Twitter", "Fox Corporation Class A", "Fox Corporation Class B","Yahoo! Inc.",
                 "Moody's Corp", "MSCI", "S&P Global, Inc.", "News Corp. Class A", "News Corp. Class B"]
#                 "Morgan Stanley", "Goldman Sachs Group", "Citigroup Inc.", "Wells Fargo", "JPMorgan Chase & Co."]
publisher_verbs = ["publish", "say", "accord", "post", "report", "tweet", "share", "write", "announce", "state", "claim", "correspond"]

"""
FUNCTIONS FOR COMPANY ALIGNMENT
"""
def initializer(keydict_path):
    # Initializer function for all the processes
    global keyword_dict, nlp, counter_threshold
    # Load the keyword dictionary
    keyword_dict = load_keyword_dict(keydict_path)
    # Load the Spacy model
    nlp = spacy.load('en_core_web_lg')

def process_row(row):
    """
    Function for multiprocessing. Processes one row
    :param row:
    :return:
    """
    # Extract data from the Pandas series
    title = row['title']
    text = row['text']
    counter = eval(row['counter'])
    aligned_companies = []

    # Function that checks that a company is present in the given text
    def check_text(text):
        """
        Simple helper function to check that any company appears in the give text
        """
        found_companies = []
        for comp in counter:
            if comp == "QuintilesIMS": # Company has been updates since due to a merger
                comp = "Quintiles and IMS Health (IQVIA)"
            if any([x in text for x in keyword_dict[comp]]):
                found_companies.append(comp)
        return found_companies

    def check_publishers(text, companies):
        """
        Rigorous testing for the published companies in the given text including dependency parsing, traversal and lemmatizing
        """
        companies_stay = set()
        doc = nlp(text, disable=['ner'])
        for chunk in doc.noun_chunks:
            # Companies that appear in the chunk
            companies_in_chunk = [comp for comp in companies_to_check for x in keyword_dict[comp] if x in chunk.text]
            #print('Companies in chunk: ', companies_in_chunk, chunk.text)
            if len(companies_in_chunk) == 0:
                continue
            # Traverse the chunk
            root = chunk.root
            traverse = [root]
            while root.dep_ != 'ROOT' and root.pos_ != 'VERB':
                root = root.head
                traverse.append(root)
            # Add if verb is not in the wrong list
            if traverse[-1].lemma_ not in publisher_verbs:
                companies_stay.update(companies_in_chunk)
            #print('Traverse: ', traverse)
        #print('Companies to stay: ', companies_stay)
        companies = [x for x in companies if (x not in companies_to_check) or (x in companies_stay)]
        #print('companies: ', companies)
        return companies

    # First check the title
    if not pd.isna(title):
        aligned_companies += check_text(title)
    if len(aligned_companies) != 0:
        # Check that the found companies are not in the "social/publishing" list
        companies_to_check = [x for x in aligned_companies if x in publisher_list]
        # If no "publisher" company then return the found one(s)
        if len(companies_to_check) == 0:
            return aligned_companies
        # Else check further
        else:
            aligned_companies = check_publishers(title, aligned_companies)
            if len(aligned_companies) != 0:
                return aligned_companies

    # Check counter
    if len(counter) > counter_threshold:
        return None

    # If the first paragraph
    paragraphs = text.split('\n')
    first_paragraph = None
    for i in range(len(paragraphs)):
        first_paragraph = paragraphs[i]
        if len(first_paragraph) == 0:
            continue
        elif any([x == first_paragraph[-1] for x in ['.', '!', '?', ':', '"', "'", ")"]]) and (
                first_paragraph[0].isalpha() or first_paragraph[0].isnumeric() or first_paragraph[0] in ['"', "'",
                                                                                                         "("]):
            break

    # If no first paragraph is found return None
    if first_paragraph is None:
        return None

    # Cut the first paragraph for the first 3 sentences
    sentences = sent_tokenize(first_paragraph)
    if len(sentences) > 1:
        first_paragraph = ' '.join(sentences[:3])
    else: # If only one paragraph is found cut for the first 10% of the words (but at least 20 words)
        words = first_paragraph.split(' ')
        first_paragraph = ' '.join(words[:max(int(len(words)*0.1), 20)])

    # Check the first paragraph for keywords
    aligned_companies = check_text(first_paragraph)
    if len(aligned_companies) != 0:
        # Check that the found companies are not in the "social/publishing" list
        companies_to_check = [x for x in aligned_companies if x in publisher_list]
        # If no "publisher" company then return the found one(s)
        if len(companies_to_check) == 0:
            return aligned_companies
        # Else check further
        else:
            aligned_companies = check_publishers(first_paragraph, aligned_companies)
            if len(aligned_companies) != 0:
                return aligned_companies
    else:
        return None

def process_df(df):
    # Input: chunk of the dataframe
    # Returns: list of lists or Nones
    aligned_companies = df.apply(process_row, axis = 1).values
    df['company'] = aligned_companies
    return df

"""
FUNCTIONS FOR DUPLICATE REMOVAL
"""
# Function for the duplicate title multiprocessing
df = None
def duplicate_init(data):
    global df
    df = data

def get_duplicate_title_indices(title):
    # Exception cases
    if title == "Business Highlights":
        return list(df.index[df['title'] == title])
    elif title == 'Home':
        return []
    # Indicies to drop
    drop_list = []
    # Subset the dataframe
    df_tmp = df[df['title'] == title]
    df_tmp_unique_text = df_tmp['text'].unique()

    # Iterate over the texts and drop duplicates
    for text in df_tmp_unique_text:
        tmp_publish_times = df_tmp['publish_date_utc'][df_tmp['text'] == text]
        min_time = tmp_publish_times.min()  # First publish time
        min_time_indices = tmp_publish_times.index[tmp_publish_times == min_time]
        remain_idx = min_time_indices[0]  # First article index
        drop_list += [idx for idx in tmp_publish_times.index if idx != remain_idx]
    del tmp_publish_times, min_time, min_time_indices, remain_idx
    df_tmp = df_tmp.drop(index = drop_list) # Drop duplicates

    # If only one article remains return, else also check fuzzy matching
    if df_tmp.shape[0] == 1:
        return drop_list

    # Fuzzy ratio distances for the remaining articles
    dist_matrix = np.zeros((df_tmp.shape[0], df_tmp.shape[0]))
    for i in range(df_tmp.shape[0]):
        for j in range(i + 1, df_tmp.shape[0]):
            d = fuzz.ratio(df_tmp['text'].iloc[i], df_tmp['text'].iloc[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    # Generate a graph with edges where d>=0.9
    dist_matrix[dist_matrix < 90] = 0
    dist_matrix[dist_matrix > 0] = 1
    # Find connected components
    n_components, component_labels = connected_components(dist_matrix, directed=False)
    # Pick only the first published article from all component
    for i in range(n_components):
        component_publish_times = df_tmp['publish_date_utc'][component_labels == i] # Subset df_tmp for component
        min_time = component_publish_times.min() # First publish time
        min_time_indices = component_publish_times.index[component_publish_times == min_time]
        remain_idx = min_time_indices[0] # First article index
        drop_list += [idx for idx in component_publish_times.index if idx != remain_idx]
    return drop_list

def remove_duplicates(df):
    print('\n----- REMOVING DUPLICATES -----')
    # Find the duplicate titles
    # TODO: Find fuzzy duplicates in titles
    duplicate_titles = df['title'].value_counts()
    duplicate_titles = duplicate_titles[duplicate_titles > 1]
    print('Number of duplicate titles: ', len(duplicate_titles))
    print('Number of articles with duplicated title: ', sum(duplicate_titles))
    print('Value counts of duplicate titles:')
    print(duplicate_titles.value_counts())

    # Find the indices of duplicate entries for each
    duplicate_titles = duplicate_titles.index
    pool = mp.Pool(initializer = duplicate_init, initargs = (df,))
    indices_to_drop = pool.map(get_duplicate_title_indices, duplicate_titles)
    pool.close()
    indices_to_drop = [l for sublist in indices_to_drop for l in sublist] # Flatten the list

    print('Number of rows removed due duplicacy: ', len(indices_to_drop))
    df = df.drop(index=indices_to_drop).reset_index(drop=True)
    print('Dataframe shape after removal: ', df.shape)
    return df

if __name__ == "__main__":
    # Parameters
    keydict_path = '../../data/sp500_list_25-02-2020.txt'
    data_path = '../../data/stage2_output.csv'
    output_path = '../../data/alignment_output.csv'
    num_rows = None
    chunk_size = 20

    # Read in the data
    df_iter = pd.read_csv(data_path,
                     nrows = num_rows,
                     index_col = 0,
                     chunksize = chunk_size)

    # Setup the multiprocessing pool
    pool = mp.Pool(initializer=initializer, initargs=(keydict_path,))
    print('Number of cores in the system: ', mp.cpu_count())

    # Apply the process_df function on all the entries of the iterator
    start_time = time.time()
    results = pool.imap(process_df, df_iter)
    output = pd.concat(results, ignore_index=True) # Concatenate outputs
    pool.close()
    print('---- OUTPUTS AFTER ALIGNMENT -----')
    print(output.head().to_string())
    print('Number of rows: ', len(output))
    print('Number of None labels: ', output['company'].apply(lambda x: x is None).sum())
    print('Number of company / article distribution: ', Counter([len(x) for x in output['company'] if x is not None]))
    print('\nTotal running time: {} seconds'.format(round(time.time() - start_time, 4)))

    # Remove rows without alignment
    output = output[output['company'].apply(lambda x: x is not None)].reset_index(drop=True)
    print('Number of rows after removing Nones: ', len(output))

    # Remove duplicates
    output = remove_duplicates(output)

    # Expand company lists
    print('\n----- EXPANDING THE COMPANY LISTS -----')
    row_indices = [i for i, x in enumerate(output['company']) for _ in range(len(x))] # Indicies to duplicate the necessary rows
    company_indices = [i for x in output['company'] for i in range(len(x))] # Indicies for the companies within the list of that row
    output = output.iloc[row_indices, :]
    output['company'] = [l[i] for l, i in zip(output['company'], company_indices)]
    print('---- OUTPUTS AFTER COMPANY ALIGNMENT PROCESSING -----')
    print(output.head().to_string())
    print('Number of rows: ', len(output))

    # Save the output list
    output.to_csv(output_path)