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
            strip_whitespace = i.strip()
            matches = re.search(regex, strip_whitespace)
            if matches is None:
                    bad_dois.add(i)
            else:
                    clean_dois.add(matches.group(1))
    return {'clean': list(clean_dois), 'removed': list(bad_dois)}

def clean_doi(doi: str):
    """_summary_

    Args:
        doi (_str_): _A string representing a DOI._

    Raises:
        ValueError: _The DOI is of the wrong string construction._

    Returns:
        _str_: _A string representing a valid DOI, stripped of whitespace._
    >>> clean_doi('abcd')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "<stdin>", line 18, in clean_doi
    ValueError: The doi provided 'abcd' is not a valid DOI.
    >>> clean_doi('10.2307/1551601\xa0 ')
    '10.2307/1551601'
    >>> clean_doi('https://dx.doi.org/10.2307/1551601')
    '10.2307/1551601'
    """
    regex = re.compile(r'.*(10\.\d{4,9}/[-.;()/:a-z0-9A-Z]+$)')
    stripped_doi = doi.strip()
    matches = re.search(regex, stripped_doi)
    if matches is None:
        raise ValueError(f"The doi provided '{doi}' is not a valid DOI.")
    else:
        return matches[1]

def clean_orcid(orcid):
    regex = re.compile(r'.*(10\.\d{4,9}/[-.;()/:a-z0-9A-Z]+$)')
    stripped_orcid = orcid.strip()
    matches = re.search(regex, stripped_orcid)
    if matches is None:
        raise ValueError(f"The ORCID provided {orcid} is not a valid ORCID.")
    else:
        return 'https://orcid.org/' + matches[1]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
