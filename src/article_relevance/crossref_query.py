import warnings
#import re
from datetime import datetime
import json
import boto3
import requests
import pandas as pd
from .s3_management import pull_s3, push_s3
from botocore.exceptions import ClientError
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
    # First, pull the set of existing DOIs
    doi_df = pull_s3(doi_store)
    metadata_df = pull_s3(metadata_store)
    # The function calls for DOIs and posts them in the S3 bucket as raw JSON.
    # It is not importing the metadata into a dataframe. That happens later.
    # It returns a list of DOIs that are in the S3 bucket.
    attempt = raw_crossref(doi_df['doi'], metadata_store)
    if verbose:
        print(f'Processed {len(attempt)} DOIs of the total {len(doi_df.index)} objects.')
    try:
        metadata = pull_s3(metadata_store)
    except ClientError as ex:
        # Error is thrown if there is no metadata file in S3 so we have to create an empty one.
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
    # Now we see how many of the DOIs we have in our primary list need to be
    # added to the metadata object:
    missing_dois = [i for i in attempt if i not in list(metadata['doi'])]
    if verbose:
        print(f'{len(doi_df.index)} DOIs passed to the function.')
        print(f'{len(missing_dois)} DOIs to be added to the metadata store.')
    # Define the keys we're using for our data object:
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
    s3 = boto3.client('s3')
    # crossref_dict is a list of processed dict objects pulled from the JSON.
    # We add to crossref_dict through the following loop:
    crossref_dict = []
    for doi in missing_dois:
        try:
            doi_key = 'dois/' + doi + '.json'
            result = s3.get_object(Bucket= metadata_store.get('Bucket'), Key= doi_key)
            # If the file is not there we'll hop to the exceptions, otherwise we
            # process the JSON object:
            response = json.loads(result['Body'].read())
            # Convert all keys to lowercase and only keep the "good" keys.
            response_json = {k.lower(): v for k, v in response['message'].items()}
            article_dict = {key: response_json.get(key, object) for key, object in good_keys.items()}
            article_dict['date'] = datetime.now()
            article_dict['valid'] = True
            crossref_dict.append(article_dict)
        except requests.exceptions.RequestException as e:
            # This generates an object that is effectively empty if we can't find the file:
            article_dict = {k: v for k, v in good_keys.items()}
            article_dict = { 'doi': doi, 'valid': False, 'date': datetime.now() }
            crossref_dict.append(article_dict)
            if verbose:
                warning_msg = f"DOI {doi} not found in CrossRef. Exception: {e}"
                warnings.warn(warning_msg, category = Warning)
        except Exception as e:
            # If there's some other form of error, let's just skip:
            if verbose:
                warnings.warn(e, category = Warning)
    # Once we're done we turn the dict into a DataFrame and append it to the metadata
    # DataFrame. We remove any duplicate DOIs.
    cross_ref_df = pd.concat([pd.DataFrame(crossref_dict), metadata],
                             ignore_index = True).drop_duplicates(subset=['doi'])
    push_s3(metadata_store, cross_ref_df, create = True, check = False)
    return cross_ref_df
