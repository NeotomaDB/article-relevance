from langdetect import detect

def enHelper(value):
    """
    Helper function to identify English observations. 
    Apply row-wise to impute missing language.
    """
     
    try:
        detect_lang = detect(value)
    except:
        detect_lang = "error"
    return detect_lang