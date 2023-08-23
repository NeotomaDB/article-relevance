
import pandas as pd
import os

def loadPQ(parquetPath):
    extension = ".parquet"
    if os.path.exists(parquetPath):
        pqFiles = [file for file in os.listdir(parquetPath) if file.endswith(extension)]
    if pqFiles:
        latestPQFile = max(pqFiles, key=lambda file: os.path.getmtime(os.path.join(parquetPath, file)))
        latestPQFilePath = os.path.join(parquetPath, latestPQFile)
        try:
            df = pd.read_parquet(latestPQFilePath)
            print(df.columns)
        except Exception as e:
            print("Parquet file not available, querying all GDD.")
    return df