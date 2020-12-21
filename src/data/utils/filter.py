import pickle
import re
from joblib import load
from src.data.utils.block import block
from src.data.utils.transformation import keywordExtraction

from selectolax.parser import HTMLParser

"""
FILTER CLASSES
"""
class custom_filter(block):
    def __init__(self, *args, **kwargs):
        super(custom_filter, self).__init__(*args, **kwargs)

    # Function that is applied on only one string
    def filter(self, input_text):
        return None

    # Apply the filter on either one string or an iterable of strings
    def filter_boolean(self, input_text):
        # facilitate the possibility of one input or an iterable input
        if type(input_text) in [str, dict]:
            return self.filter(input_text)
        elif hasattr(input_text, '__iter__'):
            return map(self.filter, input_text)

class slx_prefilter(custom_filter):

    def __init__(self, kw_path, drop_kw, **kwargs):
        super(slx_prefilter, self).__init__('slx_prefilter')
        self.kw_filter = keywordExtraction(kw_path)
        self.drop_kw = drop_kw

    def filter(self, input_text):

        html_tree = HTMLParser(input_text)
        if html_tree.body is None: return False

        # language check
        lang = html_tree.tags("html")[0].attributes.get("lang")
        english_ = False
        if lang in ['en', 'en-US', 'en-GB']:
            english_ = True
        elif lang is None:
            english_ = True
        if not english_: return False

        # extract text
        for tag in html_tree.css('script,style,meta,link,a,iframe'):
            tag.decompose()
        # for tag in html_tree.css('style'):
        #     tag.decompose()

        text = html_tree.body.text(separator='\n')

        # extract keywords
        keywords = self.kw_filter.extract(text)  # <= returns a Counter object (not serializable)

        for kw in self.drop_kw:
            keywords.pop(kw, None)
        if len(keywords) == 0: return False

        return True

class keyword_filter(custom_filter):
    """
    Checks that at least one keyword is present in the given text
    """
    def __init__(self, keyword_path, **kwargs):
        super(keyword_filter, self).__init__('keyword')
        self.extractor = keywordExtraction(keyword_path, keyword_type = 'list')

    def filter(self, input_text):
        if type(input_text) is dict:
            if 'counter' in input_text:
                extracted_keywords = input_text['counter']
            else:
                extracted_keywords = self.extractor.extract(input_text['text'])
        else:
            extracted_keywords = self.extractor.extract(input_text)
        if len(extracted_keywords) == 0:
            # no keywords detected, drop the news
            return False
        else:
            # at least one keyword detected, keep the news
            return True

class regex_language_filter(custom_filter):
    """
     English language regex sieve/pre-filter (for speed-up purposes).
    """

    PATTERN = "<html.*?lang=\"(.*?)\""

    def __init__(self, **kwargs):
        super(regex_language_filter, self).__init__('regex_lang')

    # Checks for the language in an input text (raw HTML)
    def filter(self, input_text):
        """
        Extracts lang attribute from all html tags (even the commented ones).
        Drops the news if none of the extracted lang values are english (en, en-US...).
        If no lang attribute present in the html doc, we keep the html doc.
       """
        res = re.findall(self.PATTERN, input_text)
        if res:
            return any(list(map(lambda x: "en" in x, res)))
        return True

class language_filter(custom_filter):
    """
    Passes on an input if its language is either "en" or None
    """
    def __init__(self, *args, **kwargs):
        super(language_filter, self).__init__('lang')

    # Checks the
    def filter(self, input_dict):
        assert type(input_dict) == dict # check that input is a dictionary
        lang = input_dict['language']
        if lang == 'en' or lang is None:
            return True
        else:
            return False

class date_filter(custom_filter):
    """
    Checks that the publication date is not None and it can be turned into UTC timestamp
    (i.e. not something like 3 months ago)
    """
    def __init__(self, utc_check = 'simple', **kwargs):
        assert utc_check in ['simple', 'utc', 'utc_full']
        super(date_filter, self).__init__('date')
        self.utc_check = utc_check

    def filter(self, input_dict):
        assert type(input_dict) == dict # check that input is a dictionary
        # Check that the publish date could actually be parsed to utc
        if input_dict['publish_date'] is None:
            return False
        if self.utc_check in ['utc', 'utc_full']:
            if input_dict['publish_date_utc'] is None:
                return False
        if self.utc_check == 'utc_full':
            if input_dict['publish_date_utc'].tzinfo is None:
                return False
        return True


class keyword_threshold_filter(custom_filter):
    """
    Removes the articles based on some predefined rules on the keyword counts
    """
    def __init__(self, threshold = 1, one_mentioned = False, **kwargs):
        super(keyword_threshold_filter, self).__init__('keyword_threshold_filter')
        self.threshold = threshold
        self.one_mentioned = one_mentioned

    def filter(self, input_dict, threshold = 1):
        assert type(input_dict) is dict
        counter = input_dict['counter']
        if self.one_mentioned and len(counter) != 1:
            return False
        elif len(counter) == 0:
            return False
        elif all(counter[key]  <= threshold for key in counter):
            # discard the news if  no company mentioned more than once
            return False
        else:
            return True

class finance_filter(custom_filter):
    """
    Filter class that removes non-financial articles
    """
    def __init__(self, model_path, cutoff_value = 0.5, **kwargs):
        super(custom_filter, self).__init__('finance')

        self.model_path = model_path
        self.model = load(self.model_path)
        self.cutoff_value = cutoff_value

    # Has to overwrite filter_boolean for performance
    def filter_boolean(self, input_dict):
        assert type(input_dict) is dict
        input_text = input_dict['text']
        if type(input_text) == str: # Only if one text file is feed in
            pred = self.model.predict_proba([input_text]) # Return shape (1, vocabulary_size)
            return pred[0,1]>self.cutoff_value
        elif hasattr(input_text, '__iter__'): # If an iterable is feed in
            # Length here is the number of items in the text object
            pred = self.model.predict_proba(input_text) # np.array of size (length,)
            return pred[:,1]>self.cutoff_value
