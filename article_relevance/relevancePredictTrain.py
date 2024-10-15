import joblib
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, accuracy_score

def relevancePredictTrain(x_train, y_train, classifiers):
    """
    x_train
    y_train
    classifiers (list of tuples)    List of tuples with the classifiers to try and their param grids
    """

    # Making sure x_train contains only required columns
    selected_columns = [col for col in x_train.columns if col.startswith('embedding')]
    selected_columns = selected_columns + ['doi']

    x_train = x_train[selected_columns]

    # Processing of Elements that need fit-transform
    print("Setting up features")
    preprocessor = ColumnTransformer(
        transformers = [
            ('doi', 'drop', ['doi'])],
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
    #classifier, param_grid = classifiers[0]
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
        randomized_search.fit(x_train, y_train)
        fit_time = datetime.now() - starttime
        best_classifier = randomized_search.best_estimator_
        joblib.dump(best_classifier, f"./data/models/{classifier_name}_{timestamp}.joblib")
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

    joblib.dump(megaDictionary, f"./data/results/Iteration_{timestamp}.joblib")

    print("finished process; returning results")

    return megaDictionary
