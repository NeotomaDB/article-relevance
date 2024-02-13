import warnings
#import re
from datetime import datetime
import requests
import pandas as pd
from .s3_management import pull_s3, push_s3
from .raw_crossref import raw_crossref
#from src.article_relevance import rel_print
#from .logs import get_logger

#logger = get_logger(__name__)

def crossref_query(doi_store, metadata_store, verbose = False, check = False, create = False):
    """
    Extract metadata from the Crossref API for articles in the DOI csv file.
    Extracted data are returned in a pandas dataframe.

    If a DOI is not found on CrossRef, the DOI will be in the log file. 
    
    Args:
        doi_list (list): List of DOIs
        metadata_store (s3_object): A parquet file in an Amazon S3 repository.
        verbose (boolean): Should we log outputs?
        check (boolean): Should we check against existing records?
        create (boolean): If there is no metadata store, should we create one?

    Return:
        pandas Dataframe containing CrossRef metadata.
    """
    dois = pull_s3(doi_store)
    # The function calls for DOIs and posts them in the S3 bucket as raw JSON.
    raw_crossref(dois['doi'], doi_store['Bucket'])
    try:
        metadata = pull_s3(metadata_store)
    except ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            if create:
                metadata = pd.DataFrame({'doi': [],
                                         'title': None,
                                         'subtitle': None,
                                         'author': None,
                                         'subject': None,
                                         'abstract': None,
                                         'container-title': None,
                                         'language': None,
                                         'published': None,
                                         'publisher': None,
                                         'url': None,
                                         'date': None})
    print(f'{len(dois.index)} DOIs in the data store.')
    checkers = [i for i in dois['doi'] if i not in metadata['doi']]
    print(f'{len(checkers)} DOIs to be resolved.')
    doi_list = [str(element).lower() for element in checkers]
    crossref_dict = list()
    good_keys = {'doi': '',
                 'title': [],
                 'subtitle': [],
                 'author': [],
                 'subject': [],
                 'abstract': '',
                 'container-title': [],
                 'language': '',
                 'published': {},
                 'publisher': '',
                 'url': '',
                 'valid': True}
    for doi in doi_list:
        try:
            raw_crossref(doi_list, metadata_store.get('Bucket'))
            # Convert all keys to lowercase and only keep the "good" keys.
            response_json = {k.lower(): v for k, v in response_json['message'].items()}
            article_dict = {key: response_json.get(key, object) for key, object in good_keys.items()}
            article_dict['date'] = datetime.now()
            article_dict['valid'] = True
            crossref_dict.append(article_dict)
        except requests.exceptions.RequestException as e:
            article_dict = {k: v for k, v in good_keys.items()}
            article_dict = { 'doi': doi, 'valid': False, 'date': datetime.now() }
            crossref_dict.append(article_dict)
            warning_msg = f"DOI {doi} not found in CrossRef. Exception: {e}"
            warnings.warn(warning_msg, category = Warning)
    cross_ref_df = pd.concat([pd.DataFrame(crossref_dict), metadata],
                             ignore_index = True).drop_duplicates(subset=['doi'])
    push_s3(metadata_store, cross_ref_df, create = True, check = False)
    return cross_ref_df
