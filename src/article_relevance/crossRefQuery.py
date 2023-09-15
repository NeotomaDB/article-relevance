import requests
import pandas as pd
import warnings
from datetime import datetime
from .logs import get_logger

logger = get_logger(__name__)

def crossRefQuery(doi_list):
    """
    Extract metadata from the Crossref API for article's in the DOI csv file.
    Extracted data are returned in a pandas dataframe.

    If a DOI is not found on CrossRef, the DOI will be in the log file. 
    
    Args:
        doi_list (list): List of DOIs

    Return:
        pandas Dataframe containing CrossRef metadata.
    """
    doi_list = [str(element).lower() for element in doi_list]
    doi_list = list(set(doi_list))
    
    logger.info(f'{len(doi_list)} DOIs to be queried from CrossRef')

    crossRefDict = list()
    keysToKeep = ['DOI', 'title', 'subtitle', 'author', 'subject', 
                  'abstract', 'container-title', 'language', 
                  'published', 'publisher',  'URL']
    for doi in doi_list:
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
    
    logger.info(f'CrossRef Query Finished.')
    return crossRefDF