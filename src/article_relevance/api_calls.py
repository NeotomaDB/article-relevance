from io import BytesIO
import pandas as pd
from .clean_dois import clean_dois
from .logs import get_logger
from .raw_crossref import pull_crossref
import requests
from requests.exceptions import ReadTimeout
import os
import json


def add_dois(dois):
    """_Insert each unique DOI. Do not replace existing DOIs._

    Args:
        dois (_list_): _A list of DOIs_

    Returns:
        _obj_: _An object with proerties `submitted` and `rejected`._
    >>> add_dois('abcd')
    No valid DOIs in the submitted set of values.
    []
    >>> add_dois('10.1590/s0102-69922012000200010')
    1 DOIs submitted.
    1 DOIs valid.
    doi was present: 10.1590/s0102-69922012000200010
    {'submitted': [{'doi': '10.1590/s0102-69922012000200010'}], 'rejected': [], 'inserted': [], 'present': [{'doi': '10.1590/s0102-69922012000200010'}]}
    """
    cleaned_entries = clean_dois(dois)

    valid_doi = cleaned_entries.get('clean')

    if len(valid_doi) == 0:
        print("No valid DOIs in the submitted set of values.")
        return valid_doi
    else:
        print(f'{len(cleaned_entries.get('clean')) + len(cleaned_entries.get('removed'))} unique DOIs submitted.')
        print(f'{len(valid_doi)} DOIs valid.')
        bodydata = [{'doi': i} for i in valid_doi]
        badapi = []
        goodapi = []
        presentapi = []
        
    for i in bodydata:
        try:
            outcome = requests.post('http://' + os.environ['API_HOME'] + '/v0.1/doi',
                        data = {'data': [json.dumps(i)]},
                        timeout = 10)
            if json.loads(outcome.content).get('data', 0) != 0:
                print(f'Added doi: {i.get('doi')}')
                goodapi.append(i)
            elif json.loads(outcome.content).get('message', 'oops') == "DOI already present.":
                print(f'doi was present: {i.get('doi')}')
                presentapi.append(i)
            else:
                print(f'Failed for: {i.get('doi')}\n{json.loads(outcome.content).get('message',None)}')
        except ReadTimeout as e:
            print(f'Connection failed for DOI {i}:')
            print(e)
            badapi.append(i)
        except Exception as e:
            badapi.append(i)
            print(f'Connection failed for DOI {i}:')
            print(e)

    dois = [i for i in bodydata if i['doi'] not in [j['doi'] for j in badapi]]

    return {'submitted': bodydata,
            'rejected': badapi,
            'inserted': goodapi,
            'present': presentapi}

def get_publication_metadata(doi = None):
    outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/doi',
                           params = {'doi': doi},
                            timeout = 10)
    if outcome.status_code == 200:
        pubrecords = json.loads(outcome.content).get('message')
    else:
        return None
    return pubrecords

def get_pub_for_embedding(model = 'allenai/specter2_base'):
    outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/doi/embeddingtext',
                           params = {'embeddingmodel': model},
                            timeout = 10)
    if outcome.status_code == 200:
        pubrecords = json.loads(outcome.content).get('message')
    else:
        return None
    return pubrecords

def submit_embedding(embedding_dict):
    try:
        outcome = requests.post('http://' + os.environ['API_HOME'] + '/v0.1/doi/embeddings',
                    data = {'data': json.dumps(embedding_dict, default=str)},
                    timeout = 10)
        if json.loads(outcome.content).get('status', 0) == 'success':
            print(f'Added {embedding_dict.get('model')} embeddings for doi: {embedding_dict.get('doi')}')
        else:
            print(f'Failed for: {embedding_dict.get('doi')}\n{json.loads(outcome.content).get('message',None)}')
    except ReadTimeout as e:
        print(f'Connection failed for DOI {embedding_dict.get('doi')}:')
        print(e)
    except Exception as e:
        print(f'General failure for DOI {embedding_dict.get('doi')}:')
        print(e)

def embedding_exists(doi, model):
    try:
        outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/doi/embeddings',
                            params = {'doi': doi, 'model': model},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for DOI {doi}:')
        print(e)
    except Exception as e:
        print(f'General exception for DOI {doi}:')
        print(e)

def label_exists(doi, label):
    try:
        outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/doi/labels',
                            params = {'doi': doi, 'label': label},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for DOI {doi}:')
        print(e)
    except Exception as e:
        print(f'General exception for DOI {doi}:')
        print(e)