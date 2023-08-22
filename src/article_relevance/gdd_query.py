import requests
import json
import os
import numpy as np
import pandas as pd
import re
from docopt import docopt
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, date

df = pd.read_csv('data/raw/neotoma_crossref_fixed.csv')
df = df.head(4)

df_dict = df.to_dict(orient='records')

keysToKeep = {'DOI',
        'URL',
        'abstract',
        'author',
        'container-title',
        'is-referenced-by-count', # times cited
        'language',
        'published', # datetime
        'publisher', 
        'subject', # keywords of journal
        'subtitle', # subtitle are missing sometimes
        'title'
}

for i in range(0, len(df_dict)):
    print(i)
    df_dict2 = dict()
    doi = df_dict[i]['doi']
    print(doi)
    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json()
        response_json = response_json['message']
        
        df_dict2 = {key: response_json.get(key, '') for key in keysToKeep}
        df_dict[i].update(df_dict2)
          
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}; DOI {doi} caused the error")