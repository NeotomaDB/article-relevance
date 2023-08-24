#load the parquet file
import article_relevance as ar
import pandas as pd

# Load the article elements from AWS
df = ar.loadPQ(AWS=True)

# query GDD - returns only new observations
gdd_df = ar.gddQuery(df = df, n_recent_articles = 50)

# Query new observations to crossref
crossRefDict = ar.crossRefQuery(gdd_df['DOI'])
crossRefdf = pd.DataFrame(crossRefDict)
print(crossRefdf['abstract'].head(4))
print(crossRefdf.columns)

# Process data from crossRefDict
processedData = ar.dataPreprocessing(crossRefdf)

# append new df to parquet file

