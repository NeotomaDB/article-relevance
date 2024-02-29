"""_Management of CrossRef inputs and outputs._
"""

from itertools import chain
import json
from datetime import datetime
import unicodedata
import boto3
import requests
import re
from botocore.exceptions import ClientError
import base64

def pull_crossref(doi):
    """_Pull a record from the CrossRef API and return the unstructured JSON,_
    Args:
        doi (_string_): _A DOI object._
    Returns:
        _object_: _The JSON response from CrossRef, or an smaller JSON object
                   with the exception embedded._
    """
    try:
        response = requests.get(f"https://api.crossref.org/works/{doi}",
                                        timeout = (10,10),
                                        headers = {
                    'User-Agent': 'Neotoma Publication Checker [https://github.com/NeotomaDB/article-relevance]',
                    'From': 'goring@wisc.edu'})
        response_json = response.json()
    except Exception as e:
        response_json = {'status': 'failure',
                         'message': {'DOI': doi, 
                                     'exception': str(e), 
                                     'date': str(datetime.now())}}
    return response_json


def raw_crossref(doi_list, metadata_store, verbose = False):
    """
    Extract raw Crossref JSON responses from the CrossRef API and push them to an S3 bucket
    If a DOI is not found on CrossRef, the DOI will be stored as a file with an empty JSON response.
    Args:
        doi_list (list): A list of DOIs
        metadata_bucket (s3_object): The bucket into which all DOI JSON outputs should go.
    Return:
        pandas Dataframe containing CrossRef metadata.
    """
    # First, generate a list of all valid DOIs and their associated b64 encoded file names.
    # Second, for every DOI, check that is exists in the set of json files.
    # Third, for those that don't exist, pull the CrossRef metadata.
    s3 = boto3.client('s3')
    # First find what DOIs we have in S3 as raw metadata:
    doi_stores = []
    doi_processed = []
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket = metadata_store['Bucket'], Prefix='dois')
    for page in pages:
        if page.get('KeyCount') > 0:
        # These are the DOIs that are already uploaded.
            doi_stores.append([i.get('Key') for i in page.get('Contents')])
    recovered_dois = list(chain(*doi_stores))
    print(f'DOI metadata exists for {len(recovered_dois)} records.')
    to_process = [i
                  for i in doi_list
                  if 'dois/' + base64.urlsafe_b64encode(str.encode(i)).decode('utf-8') + '.json'
                  not in recovered_dois]
    print(f'Fetching DOI metadata for {len(to_process)} records.')
    for doi in to_process:
        # Look for a file, if it's not there then poll the CrossRef API
        # If the API returns a valid response then write out the JSON response
        # If the API does not return a valid response then return an empty object
        if doi is not None:
            doi_processed.append(doi)
            doi = unicodedata.normalize('NFKD', doi).rstrip()
            filename = 'dois/' + base64.urlsafe_b64encode(str.encode(doi)).decode('utf-8') + '.json'
            newlist = [i for i in recovered_dois if i == filename]
            if len(newlist) == 0:
                if verbose:
                    print('Upload')
                    print('Polling CrossRef for object metadata.')
                response_json = pull_crossref(doi)
                if verbose:
                    if response_json['status'] == 'failure':
                        print(f'No findable DOI data for {doi}.')
                    else:
                        print(f'Recovered DOI metadata for {doi}.')
                s3.put_object(Body=json.dumps(response_json),
                            Bucket = metadata_store['Bucket'],
                            Key = filename)
    return doi_processed
