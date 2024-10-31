import joblib
from .logs import get_logger
from datetime import datetime

logger = get_logger(__name__)

def relevancePredict(processedDF,
                     model = 'logistic_regression_model.joblib',
                     predictThld = 0.5):
    """
    Make prediction on article relevance. 
    Add prediction and predict_proba to the resulting dataframe.
    Save resulting dataframe with all information in output_path directory.
    Return the resulting dataframe.

    Args:
        processedDF (pd DataFrame): Input data frame. 
        modelPath (str): Directory to trained model object.

    Returns:
        pd DataFrame with prediction and predict_proba added.
    """
    #logger.info(f'Prediction start.')
    try:
        model_object = joblib.load(model)
    except OSError:
        #logger.error("Model for article relevance not found.")
        raise(FileNotFoundError)
    model_name = model
    #logger.info(f"Running prediction for {processedDF.shape[0]} articles.")
    # Use the loaded model for prediction on a new dataset
    processedDF.loc[:, 'predict_proba'] = model_object.predict_proba(processedDF)[:, 1]
    processedDF.loc[(processedDF['predict_proba']>= predictThld), 'prediction'] = 1
    processedDF.loc[(processedDF['predict_proba']< predictThld), 'prediction'] = 0
    predictionsDF = processedDF[['doi', 'predict_proba', 'prediction']]
    predictionsDF = predictionsDF.assign(model_metadata = model_name)
    predictionsDF = predictionsDF.assign(prediction_date = datetime.now())
    #logger.info(f'Prediction completed.')
    return predictionsDF
