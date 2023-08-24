#load the parquet file
import article_relevance as ar
import pandas as pd

# Load the article elements from AWS
df = ar.loadPQ(AWS=True)
print(df.columns)

# query GDD
gdd_df = ar.gddQuery(df = df, n_recent_articles = 50)
print(gdd_df.head(3))

# query with crossref only for new articles where the abstract or other info is not existent.
crossRefDict = ar.crossRefQuery(gdd_df['DOI'])
crossRefdf=pd.DataFrame(crossRefDict)
print(crossRefdf.head(4))
print(crossRefdf.columns)

# apply the model to predict if belongs to neotoma or not

# append new df to parquet file

