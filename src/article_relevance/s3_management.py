from io import BytesIO
from datetime import datetime
import re
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from .clean_dois import clean_dois
from .logs import get_logger

def push_s3(s3_object, pa_object, check = True, create = False):
    """_Push objects to S3_
    Args:
        s3_object (_dict_): _A dict with elements `Bucket` and `Key`._
        pa_object (_DataFrame_): _A pandas DataFrame with appropriate columns._
        check (bool, optional): _Should we check for differences between source and bucket files before upload?_. Defaults to True.
        create (bool, optional): _If the bucket is missing should we create it?_. Defaults to False.
    Returns:
        _type_: _A None object if either `check` is False or if no cloud object exists._
    """
    result = None
    s3 = boto3.client('s3')
    try:
        external = s3.get_object(**s3_object)
        content = external['Body'].read()
        df = pd.read_parquet(BytesIO(content))
        if check:
            result = pa_object.compare(df)
        else:
            with BytesIO() as csv_buffer:
                pa_object.to_parquet(csv_buffer, index=False)
                try:
                    s3.put_object(Body=csv_buffer.getvalue(), **s3_object)
                except Exception as e:
                    raise e
    except ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            if create:
                print('No file currently exists with that name. Creating new object.')
                with BytesIO() as csv_buffer:
                    pa_object.to_parquet(csv_buffer, index=False)
                    try:
                        s3.put_object(Body=csv_buffer.getvalue(), **s3_object)
                    except Exception as e:
                        raise e
        else:
            raise ex
    return result

def pull_s3(s3_object):
    """_Pull an object in parquet format from the S3 bucket._

    Args:
        s3_object (_dict_): _An object with `Bucket` and `Key` keys._

    Raises:
        ex: _description_

    Returns:
        _DataFrame_: _A Pandas DataFrame._
    """
    s3 = boto3.client('s3')
    try:
        external = s3.get_object(**s3_object)
        content = external['Body'].read()
        df = pd.read_parquet(BytesIO(content))
    except ClientError as ex:
        raise ex
    return df

def update_dois(s3_object, dois, create = False):
    """_Update and manage DOI objects in the S3 bucket._

    Args:
        s3_object (_dict_): _A dict with keys "Bucket" and "Key" pointing to the location of files._
        dois (_list_): _A list of strings_

    Returns:
        _DataFrame_: _description_
    """
    valid_doi = clean_dois(dois).get('clean')
    if len(valid_doi) == 0:
        print("No valid DOIs in the submitted set of values.")
        return valid_doi
    else:
        print(f'{len(dois)} DOIs submitted.')
        print(f'{len(valid_doi)} DOIs valid.')
        try:
            df = pull_s3(s3_object)
            if not set([i for i in df.columns]) == set(['doi', 'date']):
                raise ValueError("The remote DOI file is improperly formatted.")
            else:
                df_dict = df.to_dict(orient = 'records')
                print(f'{len(df_dict)} DOI(s) already submitted.')
        except ClientError as ex:
            if ex.response['Error']['Code'] == 'NoSuchKey':
                df_dict = []
        new_doi = [{'doi': x, 'date':datetime.now()} for i, x in enumerate(valid_doi) if x not in [j.get('doi') for j in df_dict]]
        print(f'{len(new_doi)} DOIs valid and previously unseen.')
        final = pd.DataFrame(df_dict + new_doi).drop_duplicates(subset=['doi'])
        print(f'-- {len(final.index)} total DOIs.')
        push_s3(s3_object, final, check = False, create = create)
    return final
