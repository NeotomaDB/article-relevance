import re

"""_clean_dois_
>>> clean_dois("abcde")
{"clean": None, "removed": ["abcde"]}
"""

def clean_dois(dois):
    """_Clean up DOIs by removing whitespace and invalid DOIs_

    Args:
        dois (_list_): _A list of DOI values to be resolved._

    Returns:
        _list_: _A cleaned list of valid DOIs that are strings, with no leading or trailing whitespace._
    
    >>> clean_dois(["abcde"])
    {'clean': [], 'removed': ['abcde']}
    >>> clean_dois("abcde")
    {'clean': [], 'removed': ['abcde']}
    >>> clean_dois(["abcde", "10.1016/j.scitotenv.2023.163947", "https://doi.org/10.1016/j.scitotenv.2023.163947"])
    {'clean': ['10.1016/j.scitotenv.2023.163947'], 'removed': ['abcde']}
    >>> clean_dois(["abcde", ["10.1016/j.scitotenv.2023.163947", "https://doi.org/10.1016/j.scitotenv.2023.163947"]])
    An element of the doi list cannot be parsed as a string and has been removed.
    Element (<class 'list'>): ['10.1016/j.scitotenv.2023.163947', 'https://doi.org/10.1016/j.scitotenv.2023.163947']
    If you want to include the record, please fix this data element.
    {'clean': [], 'removed': ['abcde']}
    """
    if isinstance(dois, str):
        dois = [dois]
    if not isinstance(dois, list):
        dois = list(dois)
    regex = re.compile(r'.*(10\.\d{4,9}/[-.;()/:a-z0-9A-Z]+$)')
    clean_dois = set()
    bad_dois = set()
    for i in dois:
        if not isinstance(i, str):
            try:
                bad_dois.add(i)
            except TypeError as e:
                    print("An element of the doi list cannot be parsed as a string and has been removed.")
                    print(f"Element ({type(i)}): {i}")
                    print("If you want to include the record, please fix this data element.")
                    continue
        else:
            matches = re.search(regex, i)
            if matches is None:
                    bad_dois.add(i)
            else:
                    clean_dois.add(matches.group(1))
    return {'clean': list(clean_dois), 'removed': list(bad_dois)}

if __name__ == "__main__":
    import doctest
    doctest.testmod()