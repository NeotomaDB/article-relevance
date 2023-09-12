import pandas as pd
from .logs import get_logger
from sentence_transformers import SentenceTransformer,  models
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

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
    logger.info(f'Sentence embedding start.')

    #load model tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    
    #load base model
    model = AutoModel.from_pretrained('allenai/specter2_base')
    
    #load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

    valid_df = input_df[input_df["validForPrediction"] == 1]
    invalid_df = input_df[input_df["validForPrediction"] != 1]

    #text_batch = valid_df[text_col].tolist()

    # preprocess the input
    #inputs = tokenizer(text_batch, padding=True, truncation=True,
    #                    return_tensors="pt", return_token_type_ids=False, max_length=512)
    
    #output = model(**inputs)
    ###
        # Initialize an empty list to store embeddings
    all_embeddings = []
    counter = 0

    # Process text in smaller chunks
    for text in valid_df[text_col]:
        # Tokenize the text
        tokens = tokenizer.encode(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        # Pass the tokenized input through the model to obtain embeddings
        output = model(input_ids=tokens)  # Pass tokens as a dictionary

        # Take the first token in the batch as the embedding
        embeddings = output.last_hidden_state[:, 0, :].detach().numpy()

        # Append embeddings to the list
        all_embeddings.append(embeddings)

    # Stack embeddings into a numpy array
    embeddings_array = np.vstack(all_embeddings)
    ###
    # take the first token in the batch as the embedding
    #embeddings = output.last_hidden_state[:, 0, :]

    embeddings_df = pd.DataFrame(embeddings_array, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])
    #embeddings.detach().numpy()
    df_with_embeddings = pd.concat([valid_df, embeddings_df], axis = 1)
    df_with_embeddings.columns = df_with_embeddings.columns.astype(str)

    # concatenate invalid_df with valid_df
    result = pd.concat([df_with_embeddings, invalid_df])

    logger.info(f'Sentence embedding completed.')

    return result