"""_Management of CrossRef inputs and outputs._
"""

import json
from datetime import datetime
import unicodedata
import boto3
import requests
from botocore.exceptions import ClientError

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


def raw_crossref(doi_list, metadata_bucket, verbose = False):
    """
    Extract raw Crossref JSON responses from the CrossRef API and push them to an S3 bucket
    If a DOI is not found on CrossRef, the DOI will be stored as a file with an empty JSON response.
    Args:
        doi_list (list): A list of DOIs
        metadata_bucket (s3_object): The bucket into which all DOI JSON outputs should go.
    Return:
        pandas Dataframe containing CrossRef metadata.
    """
    s3 = boto3.client('s3')
    for doi in doi_list:
        # Look for a file, if it's not there then poll the CrossRef API
        # If the API returns a valid response then write out the JSON response
        # If the API does not return a valid response then return an empty object
        print(doi)
        if doi is not None:
            doi = unicodedata.normalize('NFKD', doi).rstrip()
            try:
                aa = s3.head_object(Bucket=metadata_bucket, Key=f'dois/{doi}.json')
            except Exception as ex:
                if verbose:
                    print(f'{ex}')
                response_json = pull_crossref(doi)
                aa = s3.put_object(Body=json.dumps(response_json),
                            Bucket = metadata_bucket,
                            Key = f'dois/{doi}.json')
