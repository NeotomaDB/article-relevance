import article_relevance as ar
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

### Embeddings AND Training

publicationDF = pd.read_parquet('data/parquet/publicationMetadataDF.parquet', engine='fastparquet')
annotationDF = pd.read_parquet('data/parquet/AnnotationDF.parquet')

# Create Embeddings
# Todo: Give user the choice if they want to use an embeddings file
print("Creating Embeddings, please be patient")
print(datetime.now())
bigDF = ar.addEmbeddings(publicationDF, 'titleSubtitleAbstract')
print(datetime.now())

selected_columns = [col for col in bigDF.columns if col.startswith("embedding_")]
selected_columns.append("DOI")
selected_columns.sort(key=lambda col: col != "DOI")

print("Saving embeddings")

embeddings_df = bigDF.loc[:, selected_columns]
embeddings_df.to_parquet('data/parquet/embeddingsDF.parquet', engine='fastparquet', compression='snappy', index=False)

bigDF = publicationDF.merge(embeddings_df, on = "DOI")

# This will give me a DataFrame with True Target
completeData = bigDF.merge(annotationDF, on = 'DOI')

completeData.loc[(completeData['annotation']!= 'Neotoma'), 'target'] = 0
completeData.loc[(completeData['annotation']== 'Neotoma'), 'target'] = 1
completeData.loc[(completeData['annotation']== 'Maybe Neotoma'), 'target'] = 1

## Train Starts
X = completeData.drop(columns=['DOI', 'title', 'subtitle', 'author', 'abstract',
       'language',  'URL', 'published', 'CrossRefQueryDate', 'validForPrediction', 
       'titleSubtitleAbstract', 'target', 'annotation', 'annotator', 'annotationDate', 'index'])

y = completeData['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

resultsDict = ar.relevancePredictTrain(X_train, y_train)


### Evaluating with the Test Set

# Extract all models from the joblibs, apply X_test scoring metrics or something and then get the results