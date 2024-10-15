import re

"""_clean_orcids_
>>> clean_orcids("abcde")
{"clean": None, "removed": ["abcde"]}
"""

def clean_orcids(orcids: str):
    """_Clean up ORCIDs by removing whitespace and invalid ORCIDs and ensuring they are represented as URLs_

    Args:
        orcids (_list_): _A list of ORCID values to be resolved._

    Returns:
        _list_: _A cleaned list of valid ORCIDs that are strings, with no leading or trailing whitespace._
    
    >>> clean_orcids(["abcde"])
    {'clean': [], 'removed': ['abcde']}
    >>> clean_orcids("abcde")
    {'clean': [], 'removed': ['abcde']}
    >>> clean_orcids(["abcde", "0000-0002-2700-4605", "https://orcid.org/10.1016/j.scitotenv.2023.163947"])
    {'clean': ['https://orcid.org/0000-0002-2700-4605'], 'removed': ['abcde']}
    >>> clean_orcids(["abcde", ["10.1016/j.scitotenv.2023.163947", "https://orcid.org/10.1016/j.scitotenv.2023.163947"]])
    An element of the orcid list cannot be parsed as a string and has been removed.
    Element (<class 'list'>): ['10.1016/j.scitotenv.2023.163947', 'https://orcid.org/10.1016/j.scitotenv.2023.163947']
    If you want to include the record, please fix this data element.
    {'clean': [], 'removed': ['abcde']}
    """
    if isinstance(orcids, str):
        orcids = [orcids]
    if not isinstance(orcids, list):
        orcids = list(orcids)
    regex = re.compile(r'.*([0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$)')
    clean_orcids = set()
    bad_orcids = set()
    for i in orcids:
        if not isinstance(i, str):
            try:
                bad_orcids.add(i)
            except TypeError as e:
                    print("An element of the orcid list cannot be parsed as a string and has been removed.")
                    print(f"Element ({type(i)}): {i}")
                    print("If you want to include the record, please fix this data element.")
                    continue
        else:
            matches = re.search(regex, i)
            if matches is None:
                    bad_orcids.add(i)
            else:
                    clean_orcids.add(f"https://orcid.org/{matches.group(1)}")
    return {'clean': list(clean_orcids), 'removed': list(bad_orcids)}

if __name__ == "__main__":
    import doctest
    doctest.testmod()