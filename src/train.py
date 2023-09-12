import article_relevance as ar
import pandas as pd

### Training Model

publicationDF = pd.read_parquet('data/parquet/neotomaMetadata.parquet')
annotationDF = pd.read_parquet('data/parquet/neotomaAnnotation.parquet')

model = ar.relevancePredictTrain(publicationDF, annotationDF)