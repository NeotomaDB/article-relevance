import os
from dotenv import load_dotenv
import src.article_relevance as ar
from datetime import datetime

load_dotenv()

API_HOME = os.environ['API_HOME']

processed_data = ar.get_publication_metadata()

aa = processed_data[0]

ar.label_exists(aa.get('doi'), label = 'peanut')
