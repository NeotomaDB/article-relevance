import pandas as pd
from .enHelper import enHelper
from logs import get_logger

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

    colToUse = ['subtitle', 'author', 'is-referenced-by-count', 'DOI',
       'container-title', 'language', 'URL', 'published', 'publisher', 'title',
       'abstract', 'subject', 'CrossRefQueryDate']
    
    metadataDF = metadataDF[colToUse]

    # Clean text in Subject
    metadataDF['subject'] = metadataDF['subject'].fillna(value='').apply(lambda x: ' '.join(x))

    # Clean text in title, subtitle, abstract
    metadataDF['title'] = metadataDF['title'].fillna(value='').apply(lambda x: ''.join(x))
    metadataDF['subtitle'] = metadataDF['subtitle'].fillna(value='').apply(lambda x: ''.join(x))
    #metadataDF['journal'] = metadataDF['journal'].fillna(value='').apply(lambda x: ''.join(x))

    # If an article has no abstract, consider it valid
    metadataDF.loc[(metadataDF['abstract'].isnull()), 'validForPrediction'] = 1
 
    # Remove tags from abstract
    metadataDF['abstract'] = metadataDF['abstract'].fillna(value='').apply(lambda x: ''.join(x))
    metadataDF['abstract'] = metadataDF['abstract'].str.replace(pat = '<(jats|/jats):(p|sec|title|italic|sup|sub)>', repl = ' ', regex=True)
    metadataDF['abstract'] = metadataDF['abstract'].str.replace(pat = '<(jats|/jats):(list|inline-graphic|related-article).*>', repl = ' ', regex=True)

    # Concatenate descriptive text
    metadataDF['titleSubtitleAbstract'] =  metadataDF['title'] + ' ' + metadataDF['subtitle'] + ' ' + metadataDF['abstract']

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

    print(metadataDF.head())
    '''
    keep_col = ['DOI', 'URL', 'gddid', 'valid_for_prediction', 'title_clean', 'subtitle_clean', 'abstract_clean',
                'subject_clean', 'journal', 'author', 'text_with_abstract',
                'is-referenced-by-count', 'has_abstract', 'language', 'published', 'publisher',
                'queryinfo_min_date',
                'queryinfo_max_date',
                'queryinfo_term',
                'queryinfo_n_recent']
    
    metadataDF = metadataDF.loc[:, keep_col]

    metadataDF = metadataDF.rename(columns={'title_clean': 'title',
                                'subtitle_clean': 'subtitle',
                                'abstract_clean': 'abstract'})
    
    # invalid when required input field is Null
    mask = metadataDF[['text_with_abstract', 'subject_clean', 'is-referenced-by-count', 'has_abstract']].isnull().any(axis=1)
    metadataDF.loc[mask, 'valid_for_prediction'] = 0

    mask_text = (metadataDF['text_with_abstract'].str.strip() == '')
    metadataDF.loc[mask_text, 'valid_for_prediction'] = 0

    with_missing_df = metadataDF.loc[mask, :]
    logger.info(f'{with_missing_df.shape[0]} articles has missing feature and its relevance cannot be predicted.')
    logger.info(f'Data preprocessing completed.')

    
    return metadataDF
    '''