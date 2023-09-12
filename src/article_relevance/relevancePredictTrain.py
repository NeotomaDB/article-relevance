#relevancePredictTrain
# I will not split here - we will have some CV happening just for reporting purposes. We can use a testing portion to simply use .predict(X_test) and then do the metrics elswhere
from .addEmbeddings import addEmbeddings
#from NeotomaOneHotEncoder import NeotomaOneHotEncoderTransformer
from sklearn.model_selection import RandomizedSearchCV
from sentence_transformers import SentenceTransformer


import joblib

from docopt import docopt
import json

import numpy as np
import pandas as pd
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

###
from transformers import AutoModel, AutoTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from sklearn.pipeline import Pipeline
from transformers import AutoAdapterModel

class TransformersEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, pooling_strategy="mean"):
        self.model_name = 'allenai/specter2'
        self.pooling_strategy = pooling_strategy
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

        #adapter_name = model.load_adapter("allenai/specter2", source="hf", set_active=True)

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything, so fit is a no-op
        return self

    def transform(self, X):
        # Tokenize input texts
        encoded_input = self.tokenizer(X['titleSubtitleAbstract'].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            embeddings = outputs.last_hidden_state

            # Apply pooling strategy (e.g., mean pooling)
            if self.pooling_strategy == "mean":
                embeddings = torch.mean(embeddings, dim=1)

        return embeddings.numpy()

####

def relevancePredictTrain(publicationDF, annotationDF):
    embeddedDF = addEmbeddings(publicationDF, 'titleSubtitleAbstract') #make it pull really the BT model
    completeData = embeddedDF.merge(annotationDF, on = 'DOI')

    completeData.loc[(completeData['annotation']!= 'Neotoma'), 'target'] = 0
    completeData.loc[(completeData['annotation']== 'Neotoma'), 'target'] = 1

    X = completeData.drop(columns=['DOI', 'title', 'subtitle', 'author', 'abstract',
       'language',  'URL', 'published', 'CrossRefQueryDate', 'validForPrediction', 
       'target', 'annotation', 'annotator', 'annotationDate', 'index'])
    y = completeData['target']
    
    strFeature = ['publisher']
    strTransformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')

    embedFeature = ['titleSubtitleAbstract']
    embeddingModel = TransformersEmbedding()


    preprocessor = ColumnTransformer(
        transformers = [
            ("str_preprocessor", strTransformer, strFeature),
            #("embedding_preprocessor", embeddingModel, embedFeature)
        ],
        remainder = "passthrough"
    )

    logistic_regression = LogisticRegression()

    logreg_model = make_pipeline(preprocessor, 
                                 logistic_regression)
    
    param_grid = {
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10],
    'logisticregression__max_iter': [100, 1000, 10000],
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__solver': ['liblinear', 'lbfgs']
    }
    
    # Create a randomized search with cross-validation
    randomized_search = RandomizedSearchCV(
        logreg_model, 
        param_distributions=param_grid,
        n_iter=10, 
        cv=5,
        random_state=42,  # Random seed for reproducibility
        return_train_score=True, error_score='raise'
    )

    randomized_search.fit(X,y)

    best_pipeline = randomized_search.best_estimator_
    best_params = randomized_search.best_params_
    print(best_params)


    return best_pipeline



    