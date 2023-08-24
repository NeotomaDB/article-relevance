
import pandas as pd
import os
import boto3
from io import BytesIO

def loadPQ(AWS = True, parquetPath=None):
    if AWS == True:
        extension = ".parquet"
        s3 = boto3.client('s3')   
        bucket_name = 'metareview' #load this as env variables
        object_key = 'article-relevance-output-all.parquet' #load this as env variables
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read()     
        df = pd.read_parquet(BytesIO(content))
    else:
        if parquetPath == None:
            raise ValueError("When AWS is False, a path must be provided")  
        if os.path.exists(parquetPath):
            pqFiles = [file for file in os.listdir(parquetPath) if file.endswith(extension)]
        if pqFiles:
            latestPQFile = max(pqFiles, key=lambda file: os.path.getmtime(os.path.join(parquetPath, file)))
            latestPQFilePath = os.path.join(parquetPath, latestPQFile)
            try:
                df = pd.read_parquet(latestPQFilePath)
            except Exception as e:
                print("Parquet file not available, querying all GDD.")
    return df