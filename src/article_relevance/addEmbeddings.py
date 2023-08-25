import pandas as pd
from src.logs import get_logger
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

def addEmbeddings(input_df, text_col, model = 'allenai/specter2'):
    """
    Add sentence embeddings to the dataframe using the specified model.
    
    Args:
        input_df (pd DataFrame): Input data frame. 
        text_col (str): Column with text feature.
        model(str): model name on hugging face model hub.

    Returns:
        pd DataFrame with origianl features and sentence embedding features added.
    """
    logger.info(f'Sentence embedding start.')

    embedding_model = SentenceTransformer(model)

    valid_df = input_df.query("validForPrediction == 1")
    invalid_df = input_df.query("validForPrediction != 1")

    # add embeddings to valid_df
    embeddings = valid_df[text_col].apply(embedding_model.encode)
    embeddings_df = pd.DataFrame(embeddings.tolist())
    embeddings_df.index = valid_df.index

    df_with_embeddings = pd.concat([valid_df, embeddings_df], axis = 1)
    df_with_embeddings.columns = df_with_embeddings.columns.astype(str)

    # concatenate invalid_df with valid_df
    result = pd.concat([df_with_embeddings, invalid_df])

    logger.info(f'Sentence embedding completed.')

    return result