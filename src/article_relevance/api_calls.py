from io import BytesIO
from datetime import datetime
import re
import pandas as pd
from .clean_dois import clean_dois
from .logs import get_logger
from .raw_crossref import pull_crossref
import requests 

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
        for i in bodydata:
            outcome = requests.post('http://' + API_HOME + '/doi',
                        data = {'data': json.dumps(i)},
                        timeout = 10)
            if outcome.status_code == 500:
                badapi.append(i)
    return {'submitted': dois,
            'rejected': badapi}

def add_metadata():
    outcome = requests.get('http://' + API_HOME + '/doi',
                        timeout = 10)
    if outcome.status_code == 200:
        doi_store = json.loads(outcome.content).get('message')
        for i in doi_store:
            if i.get('metadata') == 0:
                cr = pull_crossref(i.get('doi'))
                if cr.get('status') == 'ok':
                    cr_data = cr.get('message')
                    good_keys = {'DOI': '',
                        'title': [],
                        'subtitle': [],
                        'author': [],
                        'subject': [],
                        'abstract': '',
                        'container-title': [],
                        'language': '',
                        'published': {},
                        'publisher': '',
                        'URL': '',
                        'valid': True}
                    article_dict = {key.lower(): cr_data.get(key, object) for key, object in good_keys.items()}
                    
                    outcome = requests.post('http://' + API_HOME + '/doi',
                        data = {'data': json.dumps(cr)},
                        timeout = 10)
                    