from .enHelper import enHelper
from .logs import get_logger
import pandas as pd
from .s3_management import pull_s3
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
import lxml
from .api_calls import get_pub_for_embedding


#logger = get_logger(__name__)

def data_preprocessing(model_name = 'allenai/specter2_base'):
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metadata = get_pub_for_embedding(model = model_name)
    # Join arrays:
    text_batch = [{'doi': d.get('doi'),
                   'text': (d.get('title') or '').lower() + tokenizer.sep_token +
                  (d.get('subtitle') or '').lower() + tokenizer.sep_token +
                  (d.get('abstract') or '').lower(),
                  'language': d.get('language')} for d in metadata]
    # Remove HTML tags from abstract
    clean_text = [{'doi': i.get('doi'), 
                   'text': BeautifulSoup(i.get('text'), "lxml").text,
                   'language': i.get('language')} for i in text_batch]
    
    # Impute only when there are > 5 characters for langdetect to impute accurately
    for i in clean_text:
        if (i.get('language') is None) & (i.get('text') != '[SEP][SEP]') & (len(str(i.get('text') or ''))):
            i['language'] = enHelper(i.get('text'))

    return clean_text
