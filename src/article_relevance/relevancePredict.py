import pandas as pd
import joblib
import boto3
from io import BytesIO
from .logs import get_logger
from datetime import datetime

logger = get_logger(__name__)

def relevancePredict(processedDF, 
                     AWS = True, 
                     modelPath = None, 
                     predictThld = 0.5):
    """
    Make prediction on article relevancy. 
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
        object_key = 'logistic_regression_model.joblib' # change to based to the newest model, or variable to pass
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

    # split by valid for prediction
    validDF = processedDF.query('validForPrediction == 1')
    invalidDF = processedDF.query('validForPrediction != 1')

    logger.info(f"Running prediction for {validDF.shape[0]} articles.")

    # filter out rows with NaN value for embeddings (Is this necessary?)
    feature_col = [str(i) for i in range(0,768)]
    nan_exists = validDF.loc[:, feature_col].isnull().any(axis = 1)
    df_nan_exist = validDF.loc[nan_exists, :]
    validDF.loc[nan_exists, 'validForPrediction'] = 0
    logger.info(f"{df_nan_exist.shape[0]} articles's input feature contains NaN value.")

    # TODO: Retrain without has_abstract or subject_clean
    validDF['has_abstract'] = 1
    validDF.loc[(validDF['abstract'].isnull()), 'has_abstract'] = 0
    validDF['subject_clean'] = validDF['subject']
    ######## The above must be left as the .joblib file needs them but we can remove this.

    # Use the loaded model for prediction on a new dataset
    validDF.loc[:, 'predict_proba'] = model_object.predict_proba(validDF)[:, 1]
    validDF.loc[(validDF['predict_proba']>= predictThld), 'prediction'] = 1
    validDF.loc[(validDF['predict_proba']< predictThld), 'prediction'] = 0
    
    # Use 
    validDF['prediction_date'] = datetime.now()

    # Filter results, store key information that could possibly be useful downstream
    validDF = validDF[['title', 'subtitle', 'abstract',
                    'DOI', 'URL', 'validForPrediction',
                    'predict_proba', 'prediction',  'author', 
                    'language', 'published', 'publisher', 'titleSubtitleAbstract', 'prediction_date']]
    
    model_name = modelPath
    # desireable to keep from other df
    
    # Join it with invalid df to get back to the full dataframe
    result = pd.concat([validDF, invalidDF])
    result = result.reset_index()
    result['model_metadata'] = model_name
    

    logger.info(f'Prediction completed.')

    return result