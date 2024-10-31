import os
from dotenv import load_dotenv
import article_relevance as ar
import csv
import re
import pandas as pd
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier


load_dotenv()

API_HOME = os.environ['API_HOME']

with open('data/raw/neotoma_dois.csv') as file:
    db_data = list(csv.DictReader(file))

with open('data/raw/labelled_data.csv') as file:
    label_data = list(csv.DictReader(file))

all_doi = set([i.get('doi') for i in db_data] + [i.get('doi') for i in label_data])
doi_set = ar.clean_dois(all_doi)

check = ar.register_dois(all_doi)

new_dois = ['10.1590/s0102-69922012000200010', '10.1090/S0002-9939-2012-11404-2', '10.1063/1.4742131', '10.1007/s13355-012-0130-x']

check = ar.register_dois(new_dois)

processed_data = ar.data_preprocessing(model_name = 'allenai/specter2_base')

embeddings = ar.add_embeddings(processed_data, text_col = 'text', model_name = 'allenai/specter2_base')

project_exists = ar.project_exists('Neotoma Relevance')
if project_exists is None:
    ar.register_project('Neotoma Relevance', 'A project to manage models for assessing publication relevance for Neotoma.')

labels = list(set([i.get('label') for i in label_data]))
first_labels = ar.add_paper_labels(label_data, project = 'Neotoma Relevance', create = True)

neotoma_labels = [{'doi': i.get('doi'), 'label': 'In Neotoma', 'person': '0000-0002-2700-4605'} for i in db_data]

all_labels = ar.add_paper_labels(neotoma_labels, project = 'Neotoma Relevance', create = True)

# Now need to load in the labelled data and do the train/test split
data_model = ar.get_model_data(project = "Neotoma Relevance", model = "allenai/specter2_base")

data_model = [i for i in data_model if i.get('label') is not None]
data_model = [dict(item, **{'target': int(bool(re.search(pattern='Not', string=item['label'])))}) for item in data_model]
data_embedding = [i['embeddings'] for i in data_model]
data_input = pd.DataFrame(data_embedding, columns = [f'embedding_{str(i)}' for i in range(len(data_model[0]['embeddings']))])
data_input = data_input.assign(doi = [i['doi'] for i in data_model])
data_input = data_input.assign(target = [i['target'] for i in data_model])

X_train, X_test, y_train, y_test = train_test_split(data_input.copy(),
                                                    data_input['target'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=data_input['target'])

classifiers = [
    (LogisticRegression(max_iter=1000), {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [100, 1000, 10000],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    }),
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

results = ar.relevancePredict(data_input, model = 'data/models/decisiontreeclassifier_2024-09-22_22-30-35.joblib')

# Get new papers:
with open('./data/raw/newdois.csv', 'r') as file:
    new_dois = file.read().splitlines()

clean = ar.clean_dois(new_dois)
check = ar.register_dois(clean['clean'])

processed_data = ar.data_preprocessing(model_name = 'allenai/specter2_base')

embeddings = ar.add_embeddings(processed_data, text_col = 'text', model_name = 'allenai/specter2_base')

new_data_model = ar.get_model_data(project = None, model = "allenai/specter2_base")

data_embedding = [i['embeddings'] for i in new_data_model]
data_input = pd.DataFrame(data_embedding, columns = [f'embedding_{str(i)}' for i in range(len(new_data_model[0]['embeddings']))])
data_input = data_input.assign(doi = [i['doi'] for i in new_data_model])

models = [i for i in os.listdir('./data/models/') if re.match(r'^.*joblib$', i)]

results = []
for i in models:
    results.append(ar.relevancePredict(data_input, model = i))

goodpapers = results[0].loc[results[0]['prediction'] == 1]['doi'].tolist()
pubs = [ar.get_publication_metadata(i) for i in goodpapers]

counts = Counter([i[0].get('containertitle') for i in pubs])