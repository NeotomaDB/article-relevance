import re

def clean_dois(dois):
    """_Clean up DOIs by removing whitespace and invalid DOIs_

    Args:
        dois (_list_): _A list of DOI values to be resolved._

    Returns:
        _list_: _A cleaned list of valid DOIs that are strings, with no leading or trailing whitespace._
    """
    if not isinstance(dois, list):
        dois = list(dois)
    str_dois = [x.strip() for i, x in enumerate(dois) if isinstance(x, str)]
    regex = re.compile(r'^10\.\d{4,9}/[-.;()/:a-z0-9A-Z]+')
    clean_dois = [i for i in str_dois if regex.match(i)]
    bad_dois = [i for i in str_dois if i not in clean_dois]
    return {'clean': clean_dois, 'removed': bad_dois}
