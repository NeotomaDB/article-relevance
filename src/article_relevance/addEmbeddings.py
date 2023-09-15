import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
from .logs import get_logger

logger = get_logger(__name__)

def addEmbeddings(input_df, text_col):
    """
    Add sentence embeddings to the dataframe using the allenai/specter2 model.
    
    Args:
        input_df (pd DataFrame): Input data frame. 
        text_col (str): Column with text feature.
        model(str): model name on hugging face model hub.

    Returns:
        pd DataFrame with origianl features and sentence embedding features added.
    """
    logger.info(f'Starting Sentence Embedding.')

    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoModel.from_pretrained('allenai/specter2_base')
    
    # load the adapter(s) as per allenai/specter2 requirement.
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

    valid_df = input_df[input_df["validForPrediction"] == 1]
    invalid_df = input_df[input_df["validForPrediction"] != 1]

    all_embeddings = []

    # for loop to Process text in smaller chunks
    logger.info(f'Tokenizing sentences and creating embeddings')
    for text in valid_df[text_col]:
        tokens = tokenizer.encode(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        # Pass the tokenized input through the model to obtain embeddings
        output = model(input_ids=tokens)

        # Take the first token in the batch as the embedding
        embeddings = output.last_hidden_state[:, 0, :].detach().numpy()

        # Append embeddings to the list
        all_embeddings.append(embeddings)

    # Stack embeddings to an array and parse to DF
    embeddings_array = np.vstack(all_embeddings)

    embeddings_df = pd.DataFrame(embeddings_array, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])

    df_with_embeddings = pd.concat([valid_df, embeddings_df], axis = 1)

    df_with_embeddings.columns = df_with_embeddings.columns.astype(str)

    # concatenate invalid_df with valid_df
    result = pd.concat([df_with_embeddings, invalid_df])
    
    result = result.groupby('DOI', group_keys=False).apply(lambda group: group.dropna(thresh=group.notna().sum(axis=1).min()))

    logger.info(f'Sentence embedding completed.')

    return result