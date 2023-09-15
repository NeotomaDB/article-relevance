# Recreating Training Data Script
## Raw Data Files only found locally
## This script is not needed for any other purpose other than creating the initial files in AWS
## An interactive similar version can be found in the Recreating_the_Training_Set_Files.ipynb

import pandas as pd
import numpy as np
from datetime import datetime
import src.article_relevance as ar

# Log rather than print
print(datetime.now())
print("Getting Publication Metadata")
neotoma = pd.read_csv('data/raw/neotoma_crossref_fixed.csv')
pollenDF = pd.read_csv('data/raw/pollen_doc_labels.csv')
labeledDF = pd.read_csv('data/raw/project_2_labeled_data.csv')

# Remove Duplicates Based on DOI
neotoma['doi'] = neotoma['doi'].str.lower()
neotoma = neotoma.drop_duplicates(subset='doi', keep='first')

pollenDF['doi'] = pollenDF['doi'].str.lower()
pollenDF = pollenDF.drop_duplicates(subset='doi', keep='first')

labeledDF['doi'] = labeledDF['doi'].str.lower()
labeledDF = labeledDF.drop_duplicates(subset='doi', keep='first')

# CrossRef Querying
neotomaCrossRef = ar.crossRefQuery(neotoma['doi'].unique().tolist())
pollenCrossRef =  ar.crossRefQuery(pollenDF['doi'].unique().tolist())
labeledCrossRef = ar.crossRefQuery(pollenDF['doi'].unique().tolist())

# Final DF
df = pd.concat([neotomaCrossRef, pollenCrossRef, labeledCrossRef])
df = df.reset_index(drop=True)

# Basic Data Cleaning
preprocessedDF = ar.dataPreprocessing(df)

# In trainingDataSetUp the file is saved to AWS
print("Creating Publication Metadata Original Parquet")
preprocessedDF.to_parquet('data/parquet/publicationMetadataDF.parquet', 
                          engine='fastparquet', compression='snappy')

print(datetime.now())

print("Creating Annotation Data")

annotation_cols = ['DOI', 'annotation', 'annotator', 'annotationDate', 'verified', 'verifiedBy', 'verifiedTimeStamp']

# Standardizing Data Frames
print("Making different dataframes homogenous and connecting them.")
neotoma['DOI'] = neotoma['doi']
neotoma['annotation'] = 'Neotoma'
neotoma['annotator'] = 'Simon J. Goring'
neotoma['annotationDate'] = datetime.now()
neotoma['annotationDate'] = neotoma['annotationDate'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
neotoma['verified'] = 'No'
neotoma['verifiedBy'] = np.NaN
neotoma['verifiedTimeStamp'] = np.NaN
neotomaAnnotation = neotoma[annotation_cols]

pollenDF['DOI'] = pollenDF['doi']
pollenDF['annotation'] = pollenDF['Label']
pollenDF['annotator'] = pollenDF['Profile']
pollenDF['annotationDate'] = pollenDF['Timestamp']
pollenDF['verified'] = pollenDF['Verified']
pollenDF['verifiedBy'] = pollenDF['Verified By']
pollenDF['verifiedTimeStamp'] = pollenDF['Verified Timestamp']
pollenAnnotation = pollenDF[annotation_cols]

labeledDF['DOI'] = labeledDF['doi']
labeledDF['annotation'] = labeledDF['Label']
labeledDF['annotator'] = labeledDF['Profile']
labeledDF['annotationDate'] = labeledDF['Timestamp']
labeledDF['verified'] = labeledDF['Verified']
labeledDF['verifiedBy'] = labeledDF['Verified By']
labeledDF['verifiedTimeStamp'] = labeledDF['Verified Timestamp']
labeledAnnotation = labeledDF[annotation_cols]

fullAnnotation = pd.concat([neotomaAnnotation, pollenAnnotation, labeledAnnotation])
fullAnnotation = fullAnnotation.reset_index(drop=True)
fullAnnotation.head(3)

# We have to do this way because of construction; ie, with CrossRef we only take unique DOIs
fullAnnotation['DOI'] = fullAnnotation['DOI'].str.lower()
fullAnnotation = fullAnnotation.drop_duplicates(subset='DOI', keep='first')

# Shapes of ProcessingDF and AnnotationDF are not necessarily the same - 
# AnnotationsDF may have observations that were not found in crossref

print("Storing annotation in Parquet File")
fullAnnotation.to_parquet('data/parquet/AnnotationDF.parquet', index=False)

print(datetime.now())
print("Creating Annotation Data")

### With these two files we can start training.