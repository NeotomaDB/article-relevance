from .enHelper import enHelper
from src.logs import get_logger
import numpy as np

logger = get_logger(__name__)

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

    logger.info(f'Prediction data preprocessing begin.')
    
    # Clean text in Subject
    metadataDF['subject'] = metadataDF['subject'].fillna(value='').apply(lambda x: ' '.join(x))

    # Clean text in Journal
    metadataDF['container-title'] = metadataDF['container-title'].fillna(value='').apply(lambda x: ' '.join(x))

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

    metadataDF['language'] = metadataDF['language'].fillna(value = '')
    metadataDF['titleSubtitleAbstract'] = metadataDF['titleSubtitleAbstract'].fillna(value = '')

    # impute only when there are > 5 characters for langdetect to impute accurately
    imputeCondition = (metadataDF['language'].str.len() == 0) & \
                            (metadataDF['titleSubtitleAbstract'].str.contains('[a-zA-Z]', regex=True)) & \
                            (metadataDF['titleSubtitleAbstract'].str.len() >= 5)
    
    cannot_impute_condition = (metadataDF['language'].str.len() == 0) & \
                               ~((metadataDF['titleSubtitleAbstract'].str.contains('[a-zA-Z]', regex=True)) & \
                              (metadataDF['titleSubtitleAbstract'].str.len() >= 5))

    # Record info
    n_missing_lang = sum(metadataDF['language'].str.len() == 0)
    logger.info(f'{n_missing_lang} articles require language imputation')
    n_cannot_impute = sum(cannot_impute_condition)
    logger.info(f'{n_cannot_impute} cannot be imputed due to too short text metadata(title, subtitle and abstract less than 5 character).')

    # Apply imputation
    metadataDF.loc[imputeCondition,'language'] = metadataDF.loc[imputeCondition, 'titleSubtitleAbstract'].apply(lambda x: enHelper(x))

    # Set valid_for_prediction col to 0 if cannot be imputed or detected language is not English
    metadataDF.loc[(metadataDF['language']!= 'en'), 'validForPrediction'] = 0
    
    logger.info("Missing language imputation completed")
    logger.info(f"After imputation, there are {metadataDF[metadataDF['language'] != 'en'].shape[0]} non-English articles in total excluded from the prediction pipeline.")

    metadataDF.loc[(metadataDF['language']!= 'en'), 'validForPrediction'] = 0
    metadataDF.loc[(metadataDF['titleSubtitleAbstract'].isnull()), 'validForPrediction'] = 0
    metadataDF.loc[(metadataDF['subject'].isnull()), 'validForPrediction'] = 0

    # Convert author column to np.array
    metadataDF['author'] = metadataDF['author'].apply(lambda x: np.array([x]))

    return metadataDF