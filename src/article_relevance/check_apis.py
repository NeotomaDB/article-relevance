import requests
from requests.exceptions import ReadTimeout
import os
import json

def project_exists(project):
    try:
        outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/projects',
                            params = {'project': project},
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

def label_exists(label, project):
    try:
        outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/labels',
                            params = {'label': label, 'project': project},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for label: {label}:')
        print(e)
    except Exception as e:
        print(f'General exception for label: {label}:')
        print(e)

def paper_label_exists(doi, label, project, person):
    try:
        outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/doi/labels',
                            params = {'doi': doi, 'label': label, 'project': project, 'orcid': person},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for label: {label}:')
        print(e)
    except Exception as e:
        print(f'General exception for label: {label}:')
        print(e)

def embedding_exists(doi:str, model:str):
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

def person_exists(orcid):
    try:
        outcome = requests.get('http://' + os.environ['API_HOME'] + '/v0.1/people',
                            params = {'orcid': orcid},
                            timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for ORCID {orcid}:')
        print(e)
    except Exception as e:
        print(f'General exception for ORCID {orcid}:')
        print(e)

