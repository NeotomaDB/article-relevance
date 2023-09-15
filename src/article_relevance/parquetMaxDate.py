import os
from datetime import datetime
from logs import get_logger

logger = get_logger(__name__) # this gets the object with the current modules name

# See how to implement this in new file
def parquetMaxDate(parquet_path):
    """ 
    Based on the filename, find the last date when the pipeline was run.
    Return this data in yy-mm-dd format as a string.

    Args:
        parquet_path (str)      The path to the folder that stores the processed parquet files.
    
    Return:
        Date as a string.
    """
    # initialize the date with an very old date
    min_date = datetime.strptime('1800-01-01', "%Y-%m-%d").date()
    for file_name in os.listdir(parquet_path):
        file_path = os.path.join(parquet_path, file_name)
        if file_name.endswith(".parquet") and file_name.startswith('article-relevance-prediction_') and os.path.isfile(file_path):
            # Extract date from the file_name
            curr_date_str = file_name.split('_')[1][0:10]
            curr_date = datetime.strptime(curr_date_str, "%Y-%m-%d").date()
            
            # Compare with result and update if the date is newer
            if curr_date > min_date:
                 min_date = curr_date
    
    min_date_str = min_date.strftime("%Y-%m-%d")

    return min_date_str