import json

def load_keyword_json(path = 'data/sp500_list_25-02-2020.txt', companies_to_remove = []):
    """
    Loads and returns the json file containing the S&P500 companies in format
        {id: {'symbol': '', 'name': '', 'keywords': []}}
    :param path: path to json text file
    :return: dictionary
    """

    if len(companies_to_remove) == 0:
        return json.load(open(path,'r', encoding='utf-8-sig'))
    else:
        data = json.load(open(path,'r', encoding='utf-8-sig'))
        # Remove companies
        id_to_remove = [x for x in data if data[x]['name'] in companies_to_remove]
        for id in id_to_remove:
            del data[id]
        return data

def load_keyword_dict(path = 'data/sp500_list_25-02-2020.txt', companies_to_remove = []):
    """
    Returns the keyword dictionary where keys are the companies and values are the list of keywords
        {company: [keywords]}
    :param path: path to json text file
    :return: dictionary
    """
    keyword_json = load_keyword_json(path, companies_to_remove)
    keyword_dict = {}
    for key in keyword_json:
        keyword_dict[keyword_json[key]['name']] = keyword_json[key]['keywords']
    return keyword_dict

def load_all_keywords_list(path = 'data/sp500_list_25-02-2020.txt', companies_to_remove = []):
    """
    Returns a list of all the keywords in the json file
    :param path: path to json text file
    :return: list
    """
    keyword_dict = load_keyword_json(path, companies_to_remove)
    keyword_list = []
    for key in keyword_dict:
        keyword_list += keyword_dict[key]['keywords']
    return keyword_list

def load_keyword_to_company_dict(path = 'data/sp500_list_25-02-2020.txt', companies_to_remove = []):
    """
    Returns a dictionary in the form
        {'keyword' : 'company'}
    Naturally one company can occur several times in the values.
    :param path: path to json text file
    :return: dict
    """
    keyword_dict = load_keyword_dict(path, companies_to_remove)
    keyword_to_company = {}
    for comp in keyword_dict:
        for keyword in keyword_dict[comp]:
            keyword_to_company[keyword] = comp
    return keyword_to_company

def load_company_to_ticker_dict(path = 'data/sp500_list_25-02-2020.txt', companies_to_remove = []):
    """
    Returns a dictionary in the form
        {'company' : 'ticker'}
    :param path: path to json text file
    :return: dict
    """
    sp_dict = load_keyword_json(path, companies_to_remove)
    company_to_ticker = {}
    for comp in sp_dict.values():
        company_to_ticker[comp['name']] = comp['symbol']
    return company_to_ticker