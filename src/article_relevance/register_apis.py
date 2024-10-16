from io import BytesIO
import pandas as pd
from .clean_dois import clean_dois
from .logs import get_logger
from .raw_crossref import pull_crossref
import requests
from requests.exceptions import ReadTimeout
import os
import json

def register_embedding(embedding_dict):
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

def register_project(project:str, notes:str):
    """_Add a new project into the database._

    Args:
        project (str): _A short name for the project, to be used with labelling and model construction._
        notes (str): _A longer description of the project to help users understand what the project is intended to do, and who may be involved._

    Returns:
        _type_: _description_
    >>> test_register = ar.register_project("A test project", "An attempt to test.")
    """    
    try:
        outcome = requests.post('http://' + os.environ['API_HOME'] + '/v0.1/projects',
                            data = {'data': json.dumps({'project': project, 'projectnotes': notes})},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for project {project}:')
        print(e)
    except Exception as e:
        print(f'General exception for project {project}:')
        print(e)

def register_person(orcid:str):
    """_summary_

    Args:
        orcid (str): _description_

    Returns:
        _type_: _description_
    """    
    try:
        outcome = requests.post('http://' + os.environ['API_HOME'] + '/v0.1/people',
                            data = {'data': json.dumps({'person': orcid})},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for orcid {orcid}:')
        print(e)
    except Exception as e:
        print(f'General exception for orcid {orcid}:')
        print(e)


def register_label(label, project):
    try:
        outcome = requests.post('http://' + os.environ['API_HOME'] + '/v0.1/labels',
                            data = {'data': json.dumps({'project': project, 'label': label})},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for label {label}:')
        print(e)
    except Exception as e:
        print(f'General exception for label {label}:')
        print(e)

def register_paper_label(doi, label, project, orcid):
    try:
        outcome = requests.post('http://' + os.environ['API_HOME'] + '/v0.1/doi/labels',
                            data = {'data': json.dumps({'project': project.strip(), 'label': label.strip(), 'doi': doi.strip(), 'orcid': orcid.strip()})},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for label {label}:')
        print(e)
    except Exception as e:
        print(f'General exception for label {label}:')
        print(e)

def register_model(modlname: str, params: dict, model: str):
    """_Insert a model file into the database.

    Args:
        modelname (_str_): The name for the model, e.g., 'RandomForest'
        params (_dict_): A dictionary with model parameters provided.
        model (_str_): A filename for the model file location.
    """
    return None

def register_dois(dois: list, verbose: bool = True):
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

    if len(valid_doi) == 0 and verbose:
        print("No valid DOIs in the submitted set of values.")
        return valid_doi
    else:
        if verbose:
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
                if verbose:
                    print(f'Added doi: {i.get('doi')}')
                goodapi.append(i)
            elif json.loads(outcome.content).get('message', 'oops') == "DOI already present.":
                if verbose:
                    print(f'doi was present: {i.get('doi')}')
                presentapi.append(i)
            else:
                if verbose:
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
