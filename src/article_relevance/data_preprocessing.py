from .enHelper import enHelper
from .logs import get_logger
import pandas as pd
from .s3_management import pull_s3
from transformers import AutoTokenizer
from bs4 import BeautifulSoup


#logger = get_logger(__name__)

def data_preprocessing(metadata_store):
    """
    Clean up title, subtitle, abstract, subject.
    Feature engineer for descriptive text column.
    Impute language.
    The outputted dataframe is ready to be used in model prediction.
    Args:
        valid_data (pd DataFrame): Input data frame.
    Returns:
        pd DataFrame containing all info required for model prediction.
    """
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    
    metadata = pull_s3(metadata_store)
    valid_data = metadata[metadata['valid'] == True]
    # Join arrays:
    valid_data.loc[:,'title'] = [' '.join(i) for i in valid_data['title']]
    valid_data.loc[:,'subtitle'] = [' '.join(i) for i in valid_data['subtitle']]
    text_batch = [(d.get('title') or '').lower() + tokenizer.sep_token + (d.get('subtitle') or '').lower() + 
                  tokenizer.sep_token + 
                  (d.get('abstract') or '').lower() for d in valid_data.to_dict('records')]
    # Remove HTML tags from abstract
    clean_text = [BeautifulSoup(i, "lxml").text for i in text_batch]
    # Concatenate descriptive text
    valid_data.loc[:,'titleSubtitleAbstract'] = clean_text
    # Impute only when there are > 5 characters for langdetect to impute accurately
    impute_condition = (valid_data['language'].str.len() == 0) & \
                       (valid_data['titleSubtitleAbstract'].str.contains('[a-zA-Z]', regex=True)) & \
                       (valid_data['titleSubtitleAbstract'].str.len() >= 5)
    # Apply imputation
    valid_data.loc[impute_condition,'language'] = valid_data.loc[impute_condition, 'titleSubtitleAbstract'].apply(lambda x: enHelper(x))
    # Set valid_for_prediction col to 0 if cannot be imputed or detected language is not English
    valid_data.loc[(valid_data['language'] != 'en'), 'valid'] = False
    # Convert journal list to string:
    valid_data.loc[:,'container-title'] = [': '.join(i) for i in valid_data['container-title']]
    # What does this do?
    # valid_data = valid_data.groupby(valid_data.columns, axis=1).sum()
    #logger.info(f"Data Preprocessing Completed. {valid_data.shape[0]} valid observations.")
    return valid_data
