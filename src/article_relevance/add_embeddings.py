"""_Generate a dictionary of embeddings for analysis._
"""
from datetime import datetime
from itertools import chain
import pandas as pd
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from .s3_management import pull_s3, push_s3

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
    valid_df = input_df.query("valid").to_dict(orient='records')
    try:
        embedding_object = pull_s3(embedding_store).to_dict(orient='records')
    except Exception as e:
        embedding_object = []
    to_embed = [i for i in valid_df if i.get('doi') not in [j.get('doi') for j in embedding_object]]
    if len(to_embed) > 0:
        print(f'Building embeddings for {len(to_embed)} objects.')
        for i in to_embed:
            tokens = tokenizer(i.get(text_col),
                                padding='max_length',
                                truncation=True,
                                max_length=512,
                                return_tensors='pt',
                                return_token_type_ids=False)
            output = model(**tokens)
            embeddings_array = list(chain.from_iterable(output.last_hidden_state[:, 0, :].detach().numpy()))
            embeddings_key = [f"embedding_{j}" for j in range(len(embeddings_array))]
            embeddings_dict = dict(zip(embeddings_key, embeddings_array))
            embeddings_dict['doi'] = i.get('doi')
            embeddings_dict['date'] = datetime.now()
            embedding_object = embedding_object + [embeddings_dict]
        result = pd.DataFrame(embedding_object)
        push_s3(embedding_store, result, create = create, check = False)
    else:
        print('No new objects to be embedded.')
        result = embedding_object
    return result
