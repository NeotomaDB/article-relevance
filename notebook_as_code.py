import os
from dotenv import load_dotenv
import src.article_relevance as ar

load_dotenv()

DOI_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'doi_store.parquet'}
METADATA_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'metadata_store.parquet'}
EMBEDDING_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'embedding_store.parquet'}
PREDICTION_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'prediction_store.parquet'}
LABELLING_STORE =  {'Bucket':os.environ['S3_BUCKET'],'Key':'labelling_store.parquet'}

import pandas as pd
from datetime import datetime

db_data = pd.read_csv('data/raw/neotoma_dois.csv')
label_data = pd.read_csv('data/raw/labelled_data.csv')

all_doi = set(db_data['doi'].tolist() + label_data['doi'].tolist())
doi_set = ar.clean_dois(all_doi)

submitted_dois = pd.DataFrame({'doi':list(doi_set['clean']), 'date': datetime.now()})

print(f'A total of {len(db_data.index) + len(label_data.index)} DOIs were submitted.')
print(f'Of those objects there were {len(all_doi)} unique DOIs.')
print(f'There were {len(doi_set.get("clean"))} unique and valid DOIs.')

ar.push_s3(s3_object = DOI_STORE, pa_object = submitted_dois, check = False, create = True)

new_dois = ['10.1590/s0102-69922012000200010', '10.1090/S0002-9939-2012-11404-2', '10.1063/1.4742131', '10.1007/s13355-012-0130-x']

ar.update_dois(s3_object = DOI_STORE, dois = new_dois)
metadata = ar.crossref_query(DOI_STORE, METADATA_STORE, create = True)
processed_data = ar.data_preprocessing(METADATA_STORE)

input_df = processed_data
text_col = 'titleSubtitleAbstract'
model_name = 'allenai/specter2_base'
adapter_name = 'allenai/specter2'
embedding_store = EMBEDDING_STORE

embeddings = ar.add_embeddings(processed_data, 'titleSubtitleAbstract', embedding_store = EMBEDDING_STORE)

from sklearn.preprocessing import OneHotEncoder

subject_encoder = OneHotEncoder(categories='auto',
                                drop='if_binary',
                                dtype='int')

aa = subject_encoder.fit([list(i) for i in processed_data['subject'].tolist()])