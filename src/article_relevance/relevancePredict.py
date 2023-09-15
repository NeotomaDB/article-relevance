import pandas as pd
import joblib
import boto3
from io import BytesIO
from .logs import get_logger
from datetime import datetime

logger = get_logger(__name__)

def relevancePredict(processedDF, 
                     AWS = True, 
                     object_key = 'logistic_regression_model.joblib',
                     modelPath = None, 
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
    logger.info(f'Prediction start.')
    
    if AWS == True:
        s3 = boto3.client('s3')   
        bucket_name = 'metareview' #load this as env variables
        object_key = object_key
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read()     
        model_object = joblib.load(BytesIO(content))
        model_name = object_key

    else:
        if modelPath == None:
            raise ValueError("When AWS is False, a path for the model must be provided")  
        try:
        # load model
            model_object = joblib.load(modelPath)
        except OSError:
            logger.error("Model for article relevance not found.")
            raise(FileNotFoundError)

    # split by valid for prediction ???
    processedDF = processedDF[processedDF['validForPrediction'] == 1]

    logger.info(f"Running prediction for {processedDF.shape[0]} articles.")

    # filter out rows with NaN value for embeddings (Is this necessary?)
    feature_col = [str(i) for i in range(0,768)]
    nan_exists = processedDF.loc[:, feature_col].isnull().any(axis = 1)
    df_nan_exist = processedDF.loc[nan_exists, :]
    processedDF.loc[nan_exists, 'validForPrediction'] = 0
    logger.info(f"{df_nan_exist.shape[0]} articles's input feature contains NaN value.")

    # Use the loaded model for prediction on a new dataset
    processedDF.loc[:, 'predict_proba'] = model_object.predict_proba(processedDF)[:, 1]
    processedDF.loc[(processedDF['predict_proba']>= predictThld), 'prediction'] = 1
    processedDF.loc[(processedDF['predict_proba']< predictThld), 'prediction'] = 0
   
    predictionsDF = processedDF[['DOI', 'predict_proba', 'prediction']]
    predictionsDF = predictionsDF.reset_index()
    predictionsDF['model_metadata'] = model_name
    predictionsDF['prediction_date'] = datetime.now()

    logger.info(f'Prediction completed.')

    return predictionsDF