import requests
import os
import json

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
