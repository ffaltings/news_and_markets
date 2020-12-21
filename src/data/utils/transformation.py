from flashtext import KeywordProcessor
from collections import Counter
from goose3 import Goose
import pickle

from src.data.utils.block import block
from src.data.utils.keywordsIO import *
"""
Transformator object
Each inherited class of transformator reads in some data and returns data or None
"""
class transformator(block):
    def __init__(self, name):
        super(transformator, self).__init__(name)

    def transform(self, data):
        pass


class keywordExtraction(transformator):
    """
    Keyword extractor. Extracts the given list or dictionary of keywords with the Flashtext package
    """
    def __init__(self, json_path, keyword_type = 'dict', **kwargs):
        super(keywordExtraction, self).__init__('keywordExtractor')
        assert keyword_type in ['list', 'dict']
        # Load the keywords list/dict
        self.json_path = json_path

        # Setup the KeywordProcessor
        self.processor = KeywordProcessor(case_sensitive = True)
        if keyword_type == 'dict':
            self.keywords = load_keyword_dict(self.json_path)
            self.processor.add_keywords_from_dict(self.keywords)
        else:
            self.keywords = load_all_keywords_list(self.json_path)
            self.processor.add_keywords_from_list(self.keywords)

    def extract(self, text):
        """
        Extract the keywords from the input text
        :param text: input text
        :return: dict: 'keyword': number_of_occurrence
        """
        extracted_keywords = self.processor.extract_keywords(text) # Returns list of occurrences
        if type(extracted_keywords) is list:
            keyword_list = Counter(extracted_keywords)
            return dict(keyword_list)
        return dict(extracted_keywords)

    def transform(self, input_data):
        if type(input_data) is str:
            return self.extract(input_data)
        elif type(input_data) is dict:
            text = input_data['text']
            data = input_data
            data['counter'] = self.extract(text)
            if len(data['counter']) == 0:
                return None
            else:
                return data

class html_parser(transformator):
    """
    HTML parser
    returns a list: title, publish date, language, cleaned text
    """
    def __init__(self, *args, **kwargs):
        super(html_parser, self).__init__('parser')
        self.parser = Goose({'enable_image_fetching': False})

    def transform(self, data):
        """
        Parses the input text
        :param text: Either a raw html string or an iterable of those
        :return: list of (lists of) title, publication_date, meta_lang, cleaned_text
        """
        assert type(data) == str
        parsed_file = self.parser.extract(raw_html=data)
        # Just checking the publish_date_utc output
        #if parsed_file.publish_date is not None:
        #    if "\n" in parsed_file.publish_date:
        #        print('publish date')
        #        print(parsed_file.publish_date)
        #        print(parsed_file.publish_datetime_utc)
        #        print(parsed_file.publish_datetime_utc.tzinfo)
        output_dict = {'title': parsed_file.title,
                       'publish_date': parsed_file.publish_date,
                       'publish_date_utc': parsed_file.publish_datetime_utc,
                       'language': parsed_file.meta_lang,
                       'text': parsed_file.cleaned_text,
                       'url': parsed_file.final_url}

        return output_dict

class company_drop(transformator):
    """
    Drops the given list of companies from the counter
    """
    def __init__(self, company_list, **kwargs):
        super(company_drop, self).__init__('company_drop')
        self.company_list = company_list
        assert type(self.company_list) is list

    def transform(self, data):
        assert type(data) is dict

        # Remove the companies from the counter in the company_list
        counter = data['counter']
        for comp in self.company_list:
            if comp in counter:
                del counter[comp]

        # If there is no left return None
        if len(counter) == 0:
            return None
        else:
            output_data = data
            output_data['counter'] = counter
            return output_data
