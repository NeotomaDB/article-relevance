from requests import get
from requests.exceptions import ReadTimeout
import os
import json

def get_model_data(model:str, project = None):
    try:
        outcome = get(f'http://{os.environ['API_HOME']}/v0.1/modeldata',
                      params = {'project': project, 'model': model},
                      timeout = 10)
        if outcome.status_code == 200:
            call_output = json.loads(outcome.content).get('data')
            if call_output is None:
                return None
            else:
                for i in call_output:
                    i['embeddings'] = json.loads(i['embeddings'])
                return call_output
    except ReadTimeout as e:
        print(f'Connection failed for project {project}:')
        print(e)
    except Exception as e:
        print(f'General exception for project {project}:')
        print(e)
