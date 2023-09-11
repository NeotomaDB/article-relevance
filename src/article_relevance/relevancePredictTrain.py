#relevancePredictTrain
# I will not split here - we will have some CV happening just for reporting purposes. We can use a testing portion to simply use .predict(X_test) and then do the metrics elswhere
import article_relevance as ar
from NeotomaOneHotEncoder import NeotomaOneHotEncoderTransformer

import joblib

from docopt import docopt
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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

def relevancePredictTrain(publicationDF, annotationDF):
    embeddedDF = ar.addEmbeddings(publicationDF, 'titleSubtitleAbstract')
    completeData = embeddedDF.merge(annotationDF, on = 'DOI')

    completeData.loc[(completeData['annotation']!= 'Neotoma'), 'target'] = 0
    completeData.loc[(completeData['annotation']== 'Neotoma'), 'target'] = 1

    X = completeData.drop(columns=['DOI', 'title', 'subtitle', 'author', 'abstract',
       'language',  'URL', 'published', 'CrossRefQueryDate', 'validForPrediction', 
       'titleSubtitleAbstract', 'target', 'annotation', 'annotator', 'annotationDate', 'index'])
    y = completeData['target']
    
    strFeature = ['publisher']
    strTransformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    preprocessor = ColumnTransformer(
        transformers = [
            ("str_preprocessor", strTransformer, strFeature)
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
    logreg_model,  # Use the pipeline
    param_distributions=param_grid,  # Hyperparameter grid
    n_iter=10,  # Number of random combinations to try
    cv=5,  # Cross-validation folds
    random_state=42,  # Random seed for reproducibility
    return_train_score=True
    )

    randomized_search.fit(X,y)

    best_pipeline = randomized_search.best_estimator_
    best_params = randomized_search.best_params_


    return best_pipeline



    