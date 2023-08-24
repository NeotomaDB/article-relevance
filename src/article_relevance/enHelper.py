from langdetect import detect

def enHelper(value):
    ''' Helper function for en_only. 
    Apply row-wise to impute missing language data.'''
     
    try:
        detect_lang = detect(value)
    except:
        detect_lang = "error"
    return detect_lang