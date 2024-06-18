import os
from dotenv import load_dotenv
import src.article_relevance as ar

load_dotenv()

DOI_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'doi_store.parquet'}
METADATA_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'metadata_store.parquet'}
EMBEDDING_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'embedding_store.parquet'}
PREDICTION_STORE = {'Bucket':os.environ['S3_BUCKET'],'Key':'prediction_store.parquet'}
LABELLING_STORE =  {'Bucket':os.environ['S3_BUCKET'],'Key':'labelling_store.parquet'}
API_HOME = os.environ['API_HOME']

import pandas as pd
from datetime import datetime

db_data = pd.read_csv('data/raw/neotoma_dois.csv')
label_data = pd.read_csv('data/raw/labelled_data.csv')

all_doi = set(db_data['doi'].tolist() + label_data['doi'].tolist())
doi_set = ar.clean_dois(all_doi)

ar.add_dois(doi_set)



doi_output = ar.update_dois(s3_object = DOI_STORE, dois = doi_set['clean'], create = True)

print(f'A total of {len(db_data.index) + len(label_data.index)} DOIs were submitted.')
print(f'Of those objects there were {len(all_doi)} unique DOIs.')
print(f'There were {len(doi_set.get("clean"))} unique and valid DOIs.')

new_dois = ['10.1590/s0102-69922012000200010', '10.1090/S0002-9939-2012-11404-2', '10.1063/1.4742131', '10.1007/s13355-012-0130-x']

aa = ar.update_dois(s3_object = DOI_STORE, dois = new_dois)
metadata = ar.crossref_query(DOI_STORE, METADATA_STORE, create = True)
processed_data = ar.data_preprocessing(METADATA_STORE)

embeddings = ar.add_embeddings(processed_data, 'titleSubtitleAbstract', embedding_store = EMBEDDING_STORE)

first_labels = ar.add_labels(LABELLING_STORE, label_data, create = True)

neotoma_labels = pd.DataFrame([db_data.doi]).transpose().drop_duplicates()
neotoma_labels['label'] = 'Neotoma'
neotoma_labels['source'] = 'Neotoma'

all_labels = ar.add_labels(LABELLING_STORE, neotoma_labels, create = True)

# Now need to load in the labelled data and do the train/test split

dois = ar.pull_s3(DOI_STORE)
metadata = ar.pull_s3(METADATA_STORE).loc[metadata['valid'] == True]
embeddings = ar.pull_s3(EMBEDDING_STORE)
labels = ar.pull_s3(LABELLING_STORE)

data_model = pd.merge(dois,
                      metadata,
                      how = 'inner',
                      on = ['doi'])[['doi', 'valid', 'subject']].merge(embeddings.drop('date', axis = 1),
                                                                       on = ['doi'],
                                                                       how = 'inner')

labels.loc[labels['label'] == 'Neotoma', 'target'] = 1
labels.loc[labels['label'] == 'Maybe Neotoma', 'target'] = 1
labels.loc[pd.isna(labels['target']), 'target'] = 0
labels = labels.drop(['label', 'data', 'source'], axis = 1)

data_model = data_model.merge(labels, on = ['doi'], how = 'right')
data_model = data_model.loc[data_model['valid'] == True].drop(['valid','date'], axis = 1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

#data_model = data_model[data_model['subject'] != []]

X_train, X_test, y_train, y_test = train_test_split(data_model.copy(),
                                                    data_model['target'],
                                                    test_size=0.2,
                                                    random_state=42)
neotoma_encoder = ar.NeotomaOneHotEncoder(min_count=3)
X_encoded = neotoma_encoder.fit_transform(X_train[['subject']])

print(X_train.shape)
print(X_encoded.shape)
#removed_rows = neotoma_encoder.removed_rows
#print(X_train.loc[removed_rows, :])
# We want to run a set of different classifiers to determine the appropriate classification method:

classifiers = [
    (LogisticRegression(max_iter=1000), {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [100, 1000, 10000],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    })]
    (DecisionTreeClassifier(class_weight="balanced"), {
        'max_depth': range(10, 100, 10)
    }),
    (KNeighborsClassifier(weights='uniform', algorithm='auto'), {
        'n_neighbors': range(5, 100, 10)
    }),
    (BernoulliNB(binarize=0.0), {
        'alpha': [0.001, 0.01, 0.1, 1.0]
    }),
    (RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    })
]

resultsDict = ar.relevancePredictTrain(x_train = X_train, y_train = y_train, classifiers = classifiers)

with open('results.json', 'w', encoding='UTF-8') as f:
    json.dump(resultsDict['report'], f, indent=4, sort_keys=True, default=str)

