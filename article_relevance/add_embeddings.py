"""_Generate a dictionary of embeddings for analysis._
"""
from datetime import datetime
from itertools import chain
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from .check_apis import embedding_exists
from .register_apis import register_embedding

def add_embeddings(article_metadata:list,
                   text_col:str,
                   model_name:str = 'allenai/specter2_base',
                   adapter_name:str = 'allenai/specter2_classification',
                   check:bool = True,
                   register:bool = True):
    """
    Add sentence embeddings to the dataframe using the allenai/specter2 model. 
    Args:
        article_metadata (list[dict]): A list of dicts with each entry representing metadata from each article. 
        text_col (str): Coluconst doimn with text feature.
        model_name (str): Model name on hugging face model hub. Also used to index embeddings within the database.
        adapter_name (str): Adapter name on hugging face model hub.
        check (bool): Should we check to see whether an embedding for this paper & embedding currently exists in the database?
        register (bool): Should we add the embedded data to the database?
    Returns:
        list A list of embeddings for each item in article_embedding with a list of embeddings, the article doi, the date and the embedding model used.
    
    # Generate embedding on a record that does not currently exist in the database.
    # Setting `check` and `register` to `False` to support testing. 
    >>> test = ar.add_embeddings(article_metadata = [{'doi': '10.2147/dddt.s141740', 
                                               'text': 'antiglycation, radical scavenging, and semicarbazide-sensitive amine oxidase inhibitory activities of acetohydroxamic acid in vitro[SEP][SEP]',
                                               'language': 'en'}],
                       text_col = 'text',
                       model_name = 'allenai/specter2_base',
                       adapter_name = 'allenai/specter2_classification',
                       check = False,
                       register = False)
    >>> len(test)
    1
    >>> test[0].keys()
    dict_keys(['embeddings', 'doi', 'date', 'model'])
    >>> test[0].get('doi')
    '10.2147/dddt.s141740'
    >>> len(test[0].get('embeddings'))
    768
    # Try to generate an embedding without the proper article_metadata structure:
    >>> test = ar.add_embeddings(article_metadata = [{'abcd': '10.2147/dddt.s141740', 
                                               'text': 'antiglycation, radical scavenging, and semicarbazide-sensitive amine oxidase inhibitory activities of acetohydroxamic acid in vitro[SEP][SEP]',
                                               'language': 'en'}],
                       text_col = 'text',
                       model_name = 'allenai/specter2_base',
                       adapter_name = 'allenai/specter2_classification',
                       check = False,
                       register = False)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: Your article_metadata object is not consistent, either an element is missing the `doi` key, or missing the `text_col` field.
    """
    # All elements in article_metadata must be a dict, must have the key 'doi' (and a string in the field) and must
    # contain the column named in the `text_ol` column.
    
    test_fields = [all([type(i.get('doi')) is str, type(i.get(text_col)) is str]) for i in article_metadata]
    
    if not all(test_fields):
        raise ValueError("Your article_metadata object is not consistent, either an element is missing the `doi` key, or missing the `text_col` field.")
    
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
        if check:
            check_embedding = embedding_exists(doi = i.get('doi'), model = model_name)
        else:
            check_embedding = None

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
            if register:
                register_embedding(embeddings_dict)    
    assert (len(embedding_object) != len(article_metadata),
        f"The submitted object and returned object are not of the same length.")
    return embedding_object
