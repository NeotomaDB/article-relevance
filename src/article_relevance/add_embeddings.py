"""_Generate a dictionary of embeddings for analysis._
"""
from datetime import datetime
from itertools import chain
import pandas as pd
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from .api_calls import submit_embedding, embedding_exists

def add_embeddings(article_metadata, text_col,
                   model_name = 'allenai/specter2_base',
                   adapter_name = 'allenai/specter2_classification'):
    """
    Add sentence embeddings to the dataframe using the allenai/specter2 model. 
    Args:
        article_metadata (list[dict]): A list of dicts with each entry representing metadata from each article. 
        text_col (str): Coluconst doimn with text feature.
        call_remote (dict): Should the script also pull in records from the database, and push to the database?
        model_name (str): Model name on hugging face model hub. Also used to index embeddings within the database.
        adapter_name (str): Adapter name on hugging face model hub.
    Returns:
        pd DataFrame with original features and sentence embedding features added.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    # load the adapter(s) as per allenai/specter2 requirement.
    model.load_adapter(adapter_name,
                       source="hf",
                       load_as="classification",
                       set_active=True,
                       device_map='gpu')
    embedding_object = []
    print(f'Building embeddings for {len(article_metadata)} objects.')
    for i in article_metadata:
        check_embedding = embedding_exists(doi = i.get('doi'), model = model_name)
        if check_embedding is not None:
            print(f"{model_name} embeddings already exist for {i.get('doi')}.")
            embedding_object.append(check_embedding)
        else:
            tokens = tokenizer(i.get(text_col),
                                padding='max_length',
                                truncation=True,
                                max_length=512,
                                return_tensors='pt',
                                return_token_type_ids=False)
            output = model(**tokens)
            embeddings_array = list(chain.from_iterable(output.last_hidden_state[:, 0, :].detach().numpy()))
            embeddings_dict = {'embeddings': [j.item() for j in embeddings_array],
                            'doi': i.get('doi'),
                            'date': datetime.now(),
                            'model': model_name}
            embedding_object.append(embeddings_dict)
            submit_embedding(embeddings_dict)    
    assert (len(embedding_object) != len(article_metadata),
        f"The submitted object and returned object are not of the same length.")
    return embedding_object
