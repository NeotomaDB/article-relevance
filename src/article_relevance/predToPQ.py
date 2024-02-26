import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import datetime
import boto3
from io import BytesIO
from .logs import get_logger

logger = get_logger(__name__)

def predToPQ(input_df,
             AWS = True,
             object_key =  'article-relevance-output-all_00_trial_up.parquet',
             inplace = True,
             parquetPath = None):
    """
    Store prediction on article relevance in a parquet file. 
    Add prediction and predict_proba to predictionsDF
    Save resulting dataframe with all information in output_path directory.
    Return the resulting dataframe.

    Args:
        input_df (pd DataFrame): Input data frame to save as PQ. 
        model_path (str): Directory to trained model object.

    Returns:
        pd DataFrame with prediction and predict_proba added.
    """
    if AWS == True:

        ## Todo: if inplace, true, then append the new df to the old one. If false, create a new file in AWS with a timestamp

        parquet_buffer = BytesIO()
        table = pa.Table.from_pandas(input_df)
        pq.write_table(table, parquet_buffer)

        s3 = boto3.client('s3')   
        bucket_name = 'metareview'
        object_key = object_key

        parquet_buffer.seek(0)
        if inplace == True:
            # Read the existing data from S3 if it exists
            try:
                existing_data = pd.read_parquet(f's3://{bucket_name}/{object_key}')
                combined_data = pd.concat([existing_data, input_df], ignore_index=True)
                pq_buffer = BytesIO()
                table = pa.Table.from_pandas(combined_data)
                pq.write_table(table, pq_buffer)
                pq_buffer.seek(0)
                s3.upload_fileobj(pq_buffer, bucket_name, object_key)
            except:
                # If the file doesn't exist, simply upload the new data
                s3.upload_fileobj(parquet_buffer, bucket_name, object_key)
    
    else:
        if parquetPath == None:
            raise ValueError("When AWS is False, a path must be provided")  

        # ==== Save result to output_path directory =====
        parquetFolder = os.path.join(parquetPath, 'prediction_parquet')
        if not os.path.exists(parquetFolder):
            os.makedirs(parquetFolder)
        
        # Generate file name based on run date and batch
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%dT%H-%M-%S")

        # Check if a file with the current batch number already exists
        parquet_file_name = os.path.join(parquetFolder, f"article-relevance-prediction_{formatted_datetime}.parquet")
        while os.path.isfile(parquet_file_name):
            parquet_file_name = os.path.join(parquetFolder, f"article-relevance-prediction_{formatted_datetime}.parquet")

        # Write the Parquet file
        input_df.to_parquet(parquet_file_name)

    # ===== log important information ======
    logger.info(f'Total number of DOI processed: {input_df.shape[0]}')
    logger.info(f"Number of valid articles: {input_df.query('validForPrediction == 1').shape[0]}")
    logger.info(f"Number of invalid articles: {input_df.query('validForPrediction != 1').shape[0]}")
    logger.info(f"Number of relevant articles: {input_df.query('prediction == 1').shape[0]}")

    return None