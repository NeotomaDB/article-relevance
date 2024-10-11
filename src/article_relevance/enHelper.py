from langdetect import detect

def enHelper(value: str):
    """_Test to see if a string has a detectable language._

    Args:
        value (_str_): _A text string to be tested for language_

    Returns:
        _str_: _A text string indicating the language of the input, or "error" if the language can't be detected._
    
    >>> enHelper('Hello friend, how are you?')
    'en'
    >>> enHelper('12345 - 023 - 232 12')
    'error'
    >>> enHelper('portez ce vieux whisky au juge blond qui fume')
    'fr'
    >>> enHelper(None)
    'error'
    """    
    try:
        detect_lang = detect(value)
    except:
        detect_lang = "error"
    return detect_lang

if __name__ == "__main__":
    import doctest
    doctest.testmod()