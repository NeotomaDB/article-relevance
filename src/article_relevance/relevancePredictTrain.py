from .NeotomaOneHotEncoder import NeotomaOneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, accuracy_score

import joblib
from datetime import datetime

classifiers = [
        (LogisticRegression(max_iter=1000), {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'max_iter': [100, 1000, 10000],
            'penalty': ['l2']
        #  'solver': ['liblinear', 'lbfgs']
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

def relevancePredictTrain(X_train, y_train, classifiers = classifiers):
    """
    X_train
    y_train
    classifiers (list of tuples)    List of tuples with the classifiers to try and their param grids
    """

    # Making sure X_train contains only required columns
    selected_columns = [col for col in X_train.columns if col.startswith('embedding_')]
    selected_columns = selected_columns + ['subject', 'container-title']
    selected_columns.sort(key=lambda col: (col != 'subject') & (col != 'container-title'))

    X_train = X_train[selected_columns]

    # Processing of Elements that need fit-transform
    print("Setting up features")
    subFeature = ['subject', 'container-title']
    subTransformer = NeotomaOneHotEncoder(min_count=10)

    # Load the NLTK English stopwords
    nltk_stopwords = stopwords.words('english')

    # Add 'journal' to the stopwords list
    custom_stopwords = nltk_stopwords + ['book', 'journal', 'magazine']

    strFeature = 'container-title'

    strTransformer = CountVectorizer(stop_words=custom_stopwords,
                                 max_features = 100)
    
    preprocessor = ColumnTransformer(
        transformers = [
            ("str_preprocessor", strTransformer, strFeature),
            ('neotoma_encoder', subTransformer, subFeature),  
        ],
        remainder = "passthrough"
    )
    # Define the metrics you want to capture
    classification_metrics = {
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'accuracy': make_scorer(accuracy_score)
    }
    
    resultsDict = {'classifier': [],'Fit Time': [], 'train_recall': [], 'train_f1' : [],
                   'train_precision': [], 'train_accuracy': [], 'test_recall' : [],
                   'test_f1': [], 'test_precision': [], 'test_accuracy': []}
    
    megaDictionary = {'model_name': [], 'model': [], 'report': [], 'date': []}

    print("Beginning training")
    for classifier, param_grid in classifiers:
        classifier_name = str(type(classifier).__name__).lower()
        print(f"Training {classifier_name}.")

        # Define the preprocessing pipeline
        pipeline = make_pipeline(
            preprocessor,
            SimpleImputer(strategy='constant', fill_value=0), # In case there's NaNs
            classifier
        )

        param_grid = {f"{classifier_name}__{key}": value for key, value in param_grid.items()}

        randomized_search = RandomizedSearchCV(
                                estimator=pipeline,
                                param_distributions=param_grid,
                                scoring=classification_metrics,
                                cv=5,
                                n_iter=10,
                                random_state=123,
                                n_jobs=-1,
                                refit='recall',
                                return_train_score=True
                            )

        starttime = datetime.now()
        timestamp = starttime.strftime("%Y-%m-%d_%H-%M-%S")
        print(f'Starting fit at {timestamp}')

        randomized_search.fit(X_train, y_train)

        fit_time = datetime.now() - starttime

        best_classifier = randomized_search.best_estimator_
        
        joblib.dump(best_classifier, f"models/{classifier_name}_{timestamp}.joblib")
        
        best_scores_train = {
            metric: randomized_search.cv_results_[f"mean_train_{metric}"][randomized_search.best_index_]
            for metric in classification_metrics
        }

        best_scores_test = {
            metric: randomized_search.cv_results_[f"mean_test_{metric}"][randomized_search.best_index_]
            for metric in classification_metrics
        }

        classifier_name = str(type(classifier).__name__)
        resultsDict['classifier'].append(classifier_name)
        resultsDict['Fit Time'].append(fit_time)
        
        resultsDict['train_accuracy'].append(best_scores_train['accuracy'])
        resultsDict['train_precision'].append(best_scores_train['precision'])
        resultsDict['train_recall'].append(best_scores_train['recall'])
        resultsDict['train_f1'].append(best_scores_train['f1'])
        
        resultsDict['test_accuracy'].append(best_scores_test['accuracy'])
        resultsDict['test_precision'].append(best_scores_test['precision'])
        resultsDict['test_recall'].append(best_scores_test['recall'])
        resultsDict['test_f1'].append(best_scores_test['f1'])
        
        megaDictionary['model_name'].append(classifier_name)
        megaDictionary['model'].append(best_classifier)
    
    megaDictionary['report'].append(resultsDict)
    megaDictionary['date'].append(datetime.now())

    joblib.dump(megaDictionary, f"results/Iteration_{timestamp}.joblib")
    
    print("finished process; returning results")

    return megaDictionary