from io import BytesIO
from datetime import datetime
import re
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
    """

    valid_doi = clean_dois(dois).get('clean')
    if len(valid_doi) == 0:
        print("No valid DOIs in the submitted set of values.")
        return valid_doi
    else:
        print(f'{len(dois)} DOIs submitted.')
        print(f'{len(valid_doi)} DOIs valid.')
        bodydata = [{'doi': i} for i in valid_doi]
        badapi = []
        goodapi = []
        
        for i in bodydata:
            try:
                outcome = requests.post('http://' + os.environ['API_HOME'] + '/doi',
                            data = {'data': [json.dumps(i)]},
                            timeout = 10)
                if outcome.content.get('data', 0) != 0:
                    goodapi.append(i)
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
            'inserted': goodapi}

def get_pub_for_embedding(model = 'allenai/specter2_base'):
    outcome = requests.get('http://' + os.environ['API_HOME'] + '/doi/toembed',
                           params = {'embeddingmodel': model},
                            timeout = 10)
    if outcome.status_code == 200:
        pubrecords = json.loads(outcome.content).get('message')
    else:
        return None
    return pubrecords
