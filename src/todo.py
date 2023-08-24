#load the parquet file
import article_relevance as ar
import pandas as pd

# Load the article elements from AWS
df = ar.loadPQ(AWS=True)

# query GDD - returns only new observations
gdd_df = ar.gddQuery(df = df, n_recent_articles = 350)

# Query new observations to crossref
crossRefDict = ar.crossRefQuery(gdd_df['DOI'])
crossRefdf = pd.DataFrame(crossRefDict)
print(crossRefdf.shape)

# Process data from crossRefDict
processedData = ar.dataPreprocessing(crossRefdf)

# Embed data
embeddedData = ar.addEmbeddings(processedData, 'titleSubtitleAbstract')
embeddedData.to_csv('output_file_test.csv')

#Intermediate step just to gain all the needed columns:

embeddedData = embeddedData.merge(processedData, how = 'inner')
print(embeddedData.columns)

# Predict data
predictionsDF = ar.relevancePredict(embeddedData, AWS = True)
# append new df to parquet file

# Upload to parquet file
ar.predToPQ(predictionsDF.head(3), AWS = True)
