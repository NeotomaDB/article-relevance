import pandas as pd
import pyarrow.parquet as pq
import os
import re
from logs import get_logger
import requests
import json
from datetime import date
from article_relevance import loadPQ

logger = get_logger(__name__)

def gddQuery(df = None,
             n_recent_articles = None, 
             min_date = None, 
             max_date = None, 
             term = None,
             auto_check_dup = True):   
    
    # ======== Tests for input data type ==========
    if (n_recent_articles is None) and (min_date is None and max_date is None):
            raise ValueError("Either n_recent_articles or a date range should be specified.")
    if (n_recent_articles is not None) and (min_date is not None or max_date is not None):
            raise ValueError("Only one of n_recent_articles or a date range should be specified.")
    if (n_recent_articles is None) and (min_date is not None or max_date is not None):
            pattern = r'^\d{4}-\d{2}-\d{2}$'
            if min_date is not None:
                if not isinstance(min_date, str):
                     raise ValueError("min_date should be a string. min_date should be a string with format 'yyyy-mm-dd'.")
                if re.match(pattern, min_date) is False:
                     raise ValueError("min_date does not follow the correct format. min_date should be a string with format 'yyyy-mm-dd'.")
            if max_date is not None:
                if not isinstance(max_date, str):
                     raise ValueError("min_date should be a string. min_date should be a string with format 'yyyy-mm-dd'.")
                if re.match(pattern, max_date) is False:
                     raise ValueError("min_date does not follow the correct format. min_date should be a string with format 'yyyy-mm-dd'.")        
    if (n_recent_articles is not None) and (min_date is None and max_date is None):
         if not isinstance(n_recent_articles, int):
                raise ValueError("n_recent_articles should be an integer.")         
    # ========== Query API ==========
    if n_recent_articles is not None:
        logger.info(f'Querying by n_recent = {n_recent_articles}')
        api_call = "https://geodeepdive.org/api/articles?recent" + f"&max={n_recent_articles}"
    # Query API by date range
    elif (min_date is not None) and (max_date is not None):
        logger.info(f'Querying by min_date = {min_date} and max_date = {max_date}')
        api_call = f"https://xdd.wisc.edu/api/articles?min_acquired={min_date}&max_acquired={max_date}&full_results=true"
    elif (min_date is not None) and (max_date is None):
        logger.info(f'Querying by min_date = {min_date}.')
        api_call = f"https://xdd.wisc.edu/api/articles?min_acquired={min_date}&full_results=true"
    elif (min_date is None) and (max_date is not None):
        logger.info(f'Querying by max_date = {max_date}.')
        api_call = f"https://xdd.wisc.edu/api/articles?max_acquired={max_date}&full_results=true"
    else:
        raise ValueError("Please check input parameter values.")
    if term is not None:
        logger.info(f'Search term = {term}.')
        api_extend = f"&term={term}"
        api_call += api_extend
    # =========== Query xDD API to get data ==========
    session = requests.Session()
    response = session.get(api_call)
    n_refresh = 0
    while response.status_code != 200 and n_refresh < 10:
        response = requests.get(api_call)
        n_refresh += 1
    response_dict = response.json()
    data = response_dict['success']['data']
    i = 1
    logger.info(f'{len(data)} articles queried from GeoDeepDive (page {i}).')
    if 'next_page' in response_dict['success'].keys():
        next_page = response_dict['success']['next_page']
        n_refresh = 0
        while (next_page != '') :
            i += 1
            logger.info(f"going to the next page: page{i}")
            next_response = session.get(next_page)
            while next_response.status_code != 200:
                next_response = session.get(next_page)
            next_response_dict = next_response.json()
            new_data = next_response_dict['success']['data']
            logger.info(f'{len(new_data)} articles queried from GeoDeepDive (page {i}).')
            data.extend(new_data)
            next_page = next_response_dict['success']['next_page']
            n_refresh += 1
    logger.info(f'GeoDeepDive query completed.')
    # ========= Convert gdd data to dataframe =========
    # initialize the resulting dataframe
    gdd_df = pd.DataFrame()
    for article in data:
        one_article_dict = {}
        one_article_dict['gddid'] = [article['_gddid']]
        if article['identifier'][0]['type'] == 'doi':
            one_article_dict['DOI'] = [article['identifier'][0]['id']]
        else: 
            one_article_dict['DOI'] = ['Non-DOI Article ID type']
        one_article_dict['url'] = [article['link'][0]['url']]
        one_article_dict['status'] = 'queried'
        one_article = pd.DataFrame(one_article_dict)
        gdd_df = pd.concat([gdd_df, one_article])
    gdd_df = gdd_df.reset_index(drop=True)
    logger.info(f'{gdd_df.shape[0]} articles returned from GeoDeepDive.')

    # # ========== Get list of existing gddids from the parquet files =========
    if auto_check_dup == True:
        if not isinstance(df, pd.DataFrame)  or 'gddid' not in df.columns:
             raise KeyError('A data frame with gddids must be provided to check for duplicates')
        gddids = df['gddid'].unique().tolist()

        ## Filter from gdd_df all the values that already exist in the parquet:

        result_df = gdd_df[~gdd_df['gddid'].isin(gddids)]
    
        logger.info(f'{result_df.shape[0]} articles are new addition for relevance prediction.')
        
    else:
        logger.info(f'Querying all')
        result_df = gdd_df.copy()
    
    result_df['queryinfo_min_date'] = min_date
    result_df['queryinfo_max_date'] = max_date
    result_df['queryinfo_n_recent'] = n_recent_articles
    result_df['queryinfo_term'] = term
    return result_df