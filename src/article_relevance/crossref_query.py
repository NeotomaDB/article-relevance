"""_Generate and update a DataFrame of crossref data_
"""
import warnings
from datetime import datetime
import json
import base64
import boto3
import requests
import pandas as pd
from botocore.exceptions import ClientError
from .s3_management import pull_s3, push_s3
from .raw_crossref import raw_crossref
from .clean_dois import clean_dois

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
    doi_df = pull_s3(doi_store).to_dict(orient='records')
    clean_doi = clean_dois([i.get('doi') for i in doi_df])['clean']
    print(f'A total of {len(doi_df)} DOIs submitted.')
    print(f'Of the total there are {len(clean_doi)} valid DOIs.')
    raw_crossref(clean_doi, metadata_store)
    # The function calls for DOIs and posts them in the S3 bucket as raw JSON.
    # It is not importing the metadata into a dataframe. That happens later.
    # It returns a list of DOIs that are in the S3 bucket.
    try:
        metadata = pull_s3(metadata_store).to_dict(orient='records')
        print('Pulled metadata file from S3 Bucket.')
    except ClientError as ex:
        # Error is thrown if there is no metadata file in S3 so we have to create an empty one.
        print('Creating new metadata file.')
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
    missing_dois = [i for i in clean_doi if i not in [j.get('doi') for j in metadata]]
    if verbose:
        print(f'{len(clean_doi)} DOIs passed to the function.')
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
    # We add to metadata_dict through the following loop:
    if len(missing_dois) > 0:
        print(f'Pulling metadata for {len(missing_dois)} records.')
        for doi in missing_dois:
            try:
                doi_key = 'dois/' + base64.urlsafe_b64encode(str.encode(doi)).decode('utf-8') + '.json'
                result = s3.get_object(Bucket= metadata_store.get('Bucket'), Key= doi_key)
                # If the file is not there we'll hop to the exceptions, otherwise we
                # process the JSON object:
                response = json.loads(result['Body'].read())
                # Convert all keys to lowercase and only keep the "good" keys.
                response_json = {k.lower(): v for k, v in response['message'].items()}
                article_dict = {key: response_json.get(key, object) for key, object in good_keys.items()}
                article_dict['date'] = datetime.now()
                article_dict['valid'] = True
                metadata.append(article_dict)
            except requests.exceptions.RequestException as e:
                # This generates an object that is effectively empty if we can't find the file:
                print(f'Cannot find data for DOI {doi}, creating non-valid metadata entry.')
                article_dict = {k: v for k, v in good_keys.items()}
                article_dict = { 'doi': doi, 'valid': False, 'date': datetime.now() }
                metadata.append(article_dict)
                if verbose:
                    warning_msg = f"DOI {doi} not found in CrossRef. Exception: {e}"
                    warnings.warn(warning_msg, category = Warning)
            except Exception as e:
                # If there's some other form of error, let's just skip:
                if verbose:
                    warnings.warn(e, category = Warning)
        # Once we're done we turn the dict into a DataFrame and append it to the metadata
        # DataFrame. We remove any duplicate DOIs.
        cross_ref_df = pd.DataFrame(metadata).drop_duplicates(subset=['doi'])
        push_s3(metadata_store, cross_ref_df, create = create, check = False)
    else:
        print('No records to add.')
        cross_ref_df = pd.DataFrame(metadata)
    return cross_ref_df
