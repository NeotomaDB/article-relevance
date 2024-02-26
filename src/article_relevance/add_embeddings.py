import pandas as pd
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np
from .s3_management import pull_s3, push_s3
from datetime import datetime

# from .logs import get_logger
# logger = get_logger(__name__)

def add_embeddings(input_df, text_col,
                   embedding_store,
                   model_name = 'allenai/specter2_base', 
                   adapter_name = 'allenai/specter2_classification',
                   create = True):
    """
    Add sentence embeddings to the dataframe using the allenai/specter2 model. 
    Args:
        input_df (pd DataFrame): Input data frame. 
        text_col (str): Column with text feature.
        model(str): model name on hugging face model hub.
    Returns:
        pd DataFrame with original features and sentence embedding features added.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    # load the adapter(s) as per allenai/specter2 requirement.
    model.load_adapter(adapter_name,
                       source="hf",
                       load_as="classification",
                       set_active=True)
    valid_df = input_df.query("valid")
    try:
        embedding_object = pull_s3(embedding_store)
    except:
        embedding_object = pd.DataFrame({'doi':[], 'date':[]})
    for i in [i for i in valid_df['doi'] if i not in embedding_object['doi']]:
        tokens = tokenizer(valid_df.query("doi == @i")[text_col].to_list()[0],
                              padding='max_length',
                                  truncation=True,
                                  max_length=512,
                                  return_tensors='pt',
                                  return_token_type_ids=False)
        output = model(**tokens)
        embeddings_array = output.last_hidden_state[:, 0, :].detach().numpy()
        embeddings_df = pd.DataFrame(embeddings_array, columns=[f"embedding_{j}" for j in range(embeddings_array.shape[1])])
        embed_df = pd.concat([pd.DataFrame({'doi': [i]}), embeddings_df, pd.DataFrame({'date':[datetime.now()]})], axis = 1)
        embedding_object = pd.concat([embedding_object, embed_df])
    result = embedding_object.groupby('doi', group_keys=False).apply(lambda group: group.dropna(thresh=group.notna().sum(axis=1).min()))
    push_s3(embedding_store, result, create = create, check = False)
    # logger.info(f'Sentence embedding completed.')
    return result
