from .enHelper import enHelper
from .logs import get_logger
import pandas as pd
from .s3_management import pull_s3
from transformers import AutoTokenizer
from bs4 import BeautifulSoup


#logger = get_logger(__name__)

def data_preprocessing(metadata):
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

    # Join arrays:
    text_batch = [{'doi': d.get('doi'),
                   'text': (d.get('title') or '').lower() + tokenizer.sep_token +
                  (d.get('subtitle') or '').lower() + tokenizer.sep_token +
                  (d.get('abstract') or '').lower(),
                  'language': d.get('lang')} for d in metadata]
    # Remove HTML tags from abstract
    clean_text = [{'doi': i.get('doi'), 
                   'text': BeautifulSoup(i.get('text'), "lxml").text,
                   'lang': i.get('lang)')} for i in text_batch]
    
    # Impute only when there are > 5 characters for langdetect to impute accurately
    clean_text.update('impute') = [(i.get('language') is None) &
                                   (i.get('text') != '[SEP][SEP]') &
                        (len(str(i.get('text') or ''))) for i in clean_text]
    
    # Apply imputation
    valid_data.loc[impute_condition,'language'] = valid_data.loc[impute_condition, 'titleSubtitleAbstract'].apply(lambda x: enHelper(x))
    # Set valid_for_prediction col to 0 if cannot be imputed or detected language is not English
    valid_data.loc[(valid_data['language'] != 'en'), 'valid'] = False
    valid_data.loc[valid_data['subject'].apply(len) == 0, 'valid'] = False
    # Convert journal list to string:
    valid_data.loc[:,'container-title'] = [': '.join(i) for i in valid_data['container-title']]
    # What does this do?
    # valid_data = valid_data.groupby(valid_data.columns, axis=1).sum()
    #logger.info(f"Data Preprocessing Completed. {valid_data.shape[0]} valid observations.")
    return valid_data
