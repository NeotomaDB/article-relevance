# File to predict using the given models and PQ data
#load the parquet file
import article_relevance as ar
import pandas as pd

# Load the article elements from AWS
df = ar.loadPQ(AWS=True)

# query GDD - returns only new observations
gdd_df = ar.gddQuery(df = df, n_recent_articles = 10)

# Query new observations to crossref
crossRefDF = ar.crossref_query(gdd_df['DOI'])

# Process data from crossRefDict
processedData = ar.data_preprocessing(crossRefDF)

# Embed data
embeddedData = ar.add_embeddings(processedData, 'titleSubtitleAbstract')
embeddedData.to_csv('output_file_test.csv')

# Predict data
predictionsDF = ar.relevancePredict(embeddedData, AWS = True)
# append new df to parquet file

# TODO
# The information from GDD might be relevant - explore if any part is needed?

# Upload to parquet file
ar.predToPQ(predictionsDF.head(3), AWS = True)
