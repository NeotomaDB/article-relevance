import requests
import pandas as pd
import warnings
from datetime import datetime

## Todo add environ variable to use email
def crossRefQuery(doi_list):
    doi_list = [str(element).lower() for element in doi_list]
    crossRefDict = list()
    keysToKeep = ['DOI', 'title', 'subtitle', 'author', 'subject', 
                  'abstract', 'container-title', 'language', 
                  'published', 'publisher',  'URL']
    for doi in list(set(doi_list)):
        url = f"https://api.crossref.org/works/{doi}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            response_json = response.json()
            response_json = response_json['message']      
            articleDict = {key: response_json.get(key, '') for key in keysToKeep}
            articleDict['CrossRefQueryDate'] = datetime.now()
            crossRefDict.append(articleDict)        
        except requests.exceptions.RequestException as e:
            warning_msg = f"DOI {doi} not found in CrossRef"
            warnings.warn(warning_msg, category=Warning)
    crossRefDF = pd.DataFrame(crossRefDict)


    return crossRefDF