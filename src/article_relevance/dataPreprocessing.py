from .enHelper import enHelper
from .logs import get_logger
import pandas as pd

logger = get_logger(__name__)

def replace_missing_with_empty_list(value):
    if isinstance(value, list):
        return value
    return []

def dataPreprocessing(metadataDF):
    """
    Clean up title, subtitle, abstract, subject.
    Feature engineer for descriptive text column.
    Impute language.
    The outputted dataframe is ready to be used in model prediction.
    
    Args:
        metadataDF (pd DataFrame): Input data frame. 

    Returns:
        pd DataFrame containing all info required for model prediction.
    """

    logger.info(f'Data cleaning and parsing begins.')
    
    # Clean text in title, subtitle, abstract
    metadataDF['title'] = metadataDF['title'].fillna(value='').apply(lambda x: ''.join(x))
    metadataDF['subtitle'] = metadataDF['subtitle'].fillna(value='').apply(lambda x: ''.join(x))

    # If an article has no abstract, consider it valid
    metadataDF['validForPrediction'] = 1 # All articles start off as valid
    
    # Remove tags from abstract
    metadataDF['abstract'] = metadataDF['abstract'].fillna(value='').apply(lambda x: ''.join(x))
    metadataDF['abstract'] = metadataDF['abstract'].str.replace(pat = '<(jats|/jats):(p|sec|title|italic|sup|sub)>', repl = ' ', regex=True)
    metadataDF['abstract'] = metadataDF['abstract'].str.replace(pat = '<(jats|/jats):(list|inline-graphic|related-article).*>', repl = ' ', regex=True)

    # Concatenate descriptive text
    metadataDF['titleSubtitleAbstract'] =  metadataDF['title'] + ' ' + metadataDF['subtitle'] + ' ' + metadataDF['abstract']
    metadataDF['titleSubtitleAbstract'] = metadataDF['titleSubtitleAbstract'].str.lower()

    # Impute missing language
    logger.info(f'Running article language imputation.')

    # Missing subject
    metadataDF['subject'] = metadataDF['subject'].apply(replace_missing_with_empty_list)

    metadataDF['language'] = metadataDF['language'].fillna(value = '')
    metadataDF['titleSubtitleAbstract'] = metadataDF['titleSubtitleAbstract'].fillna(value = '')
    metadataDF['titleSubtitleAbstract'] = metadataDF['titleSubtitleAbstract'].str.lower()

    # Impute only when there are > 5 characters for langdetect to impute accurately
    imputeCondition = (metadataDF['language'].str.len() == 0) & \
                      (metadataDF['titleSubtitleAbstract'].str.contains('[a-zA-Z]', regex=True)) & \
                      (metadataDF['titleSubtitleAbstract'].str.len() >= 5)
    
    cannotImputeCondition = (metadataDF['language'].str.len() == 0) & \
                            ~((metadataDF['titleSubtitleAbstract'].str.contains('[a-zA-Z]', regex=True)) & \
                            (metadataDF['titleSubtitleAbstract'].str.len() >= 5))

    # Record info
    nMissingLanguage = sum(metadataDF['language'].str.len() == 0)
    logger.info(f'{nMissingLanguage} articles require language imputation')

    nCannotImpute = sum(cannotImputeCondition)
    logger.info(f'{nCannotImpute} cannot be imputed due to too short text metadata (title, subtitle and abstract are less than 5 characters).')

    # Apply imputation
    metadataDF.loc[imputeCondition,'language'] = metadataDF.loc[imputeCondition, 'titleSubtitleAbstract'].apply(lambda x: enHelper(x))

    # Set valid_for_prediction col to 0 if cannot be imputed or detected language is not English
    metadataDF.loc[(metadataDF['language']!= 'en'), 'validForPrediction'] = 0
    
    logger.info("Missing language imputation completed")
    logger.info(f"After imputation, there are {metadataDF[metadataDF['language'] != 'en'].shape[0]} non-English articles in total excluded from the prediction pipeline.")

    metadataDF.loc[(metadataDF['language']!= 'en'), 'validForPrediction'] = 0
    metadataDF.loc[(metadataDF['titleSubtitleAbstract'].isnull()), 'validForPrediction'] = 0

    # Convert author, date column to str so that we can save the parquet file
    metadataDF['author'] = metadataDF['author'].apply(str)
    metadataDF['published'] = metadataDF['published'].apply(str)

    # Convert journal list to string:
    metadataDF['container-title'] = metadataDF['container-title'].fillna(value='').apply(lambda x: ''.join(x))

    metadataDF = metadataDF.groupby(metadataDF.columns, axis=1).sum()

    #metadataDF = metadataDF[metadataDF['validForPrediction'] == 1]

    logger.info(f"Data Preprocessing Completed. {metadataDF.shape[0]} valid observations.")

    return metadataDF