class block(object):
    FILTERS = ["regex_lang", "keyword", "keyword_threshold_filter", "lang", "date", "finance", "slx_prefilter"]
    TRANSFORMATIONS = ["keywordExtractor", "parser", "company_drop"]

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.type = 'FILTER' if name in block.FILTERS else 'TRANSFORMATION'