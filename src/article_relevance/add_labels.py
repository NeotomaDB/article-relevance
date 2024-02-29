from datetime import datetime
import pandas as pd
from .s3_management import pull_s3, push_s3

def add_labels(label_store: dict, label_df: pd.DataFrame, source: str = None, create: bool = False):
    """_Add label data for DOIs in the set of metadata. Allows the user to pass in a `source`._

    Args:
        label_store (dict): _A dict with elements `Bucket` and `Key` to identify the file source._
        label_df (pd.DataFrame): _A pd.DataFrame with columns `doi` and `label` (optional `source`)._
        source (str, optional): _The label source, for example "From DB" or "Simon Goring"_. Defaults to None.
        create (bool, optional): _If no label data exists, should it be created in the cloud?_. Defaults to False.

    Returns:
        _DataFrame_: _A Pandas DataFrame with columns `doi`, `label`, `source` and `date`._
    """
    try:
        labels_dict = pull_s3(label_store).to_dict(orient='records')
    except:
        labels_dict = []
    # Each dict element should have 'doi', 'label', 'source' and 'date'.
    label_input_dict = label_df.to_dict(orient='records')
    for i in label_input_dict:
        assert all([j in i.keys() for j in ['doi', 'label']]), 'Label DataFrame must contain elements `doi` and `label`.'
        i['date'] = datetime.now()
        if 'source' not in i.keys():
            i['source'] = source
        labels_dict = labels_dict + [i]
    result = pd.DataFrame(labels_dict).drop_duplicates(
        subset = ['doi', 'source', 'label'],
        keep = 'last')
    push_s3(label_store, result, create = create, check = False)
    return result
