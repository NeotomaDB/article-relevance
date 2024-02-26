import pandas as pd
from .crossref_query import crossref_query
from .data_preprocessing import data_preprocessing
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from io import BytesIO

aws_keys = {'bucket_name': 'metareview',
            'annotation_key': 'AnnotationDF.parquet',
            'publication_key': 'publicationMetadataDF.parquet'}

def updateSource(csv_file, aws_keys=aws_keys):
    s3 = boto3.client('s3')   

    # Load annotation.pq from ASW S3 instance
    response = s3.get_object(Bucket = aws_keys['bucket_name'], 
                             Key = aws_keys['annotation_key'])
    
    content = response['Body'].read()   
    annotation_df = pd.read_parquet(BytesIO(content))
    
    # Load the CSV file
    new_annotations = pd.read_csv(csv_file)

    # Concatenate the two data frames
    combined_annotations = pd.concat([annotation_df, new_annotations], 
                                     ignore_index=True)

    # Sort the combined data frame by DOI and Date
    combined_annotations.sort_values(by=['DOI', 'annotationDate'], ascending=[True, False], 
                                     inplace=True)

    # Drop duplicates, keeping the first occurrence (which has the newest date)
    combined_annotations.drop_duplicates(subset=['DOI'], keep='first', inplace=True)

    # Reset the index of the resulting data frame
    combined_annotations.reset_index(drop=True, inplace=True)

    # Save the updated annotation.pq file
    pq_buffer = BytesIO()
    table = pa.Table.from_pandas(combined_annotations)
    
    # To Remove: Create an empty object in the specified location (key) in the S3 bucket 
    s3.put_object(Bucket=aws_keys['bucket_name'], Key='trial_annotationupdate.parquet') #remove this line
    pq.write_table(table, pq_buffer)
    #fastparquet.write(pq_buffer, table, compression='SNAPPY')
    pq_buffer.seek(0)
    
    s3.upload_fileobj(pq_buffer, 
                      Bucket = aws_keys['bucket_name'], 
                      Key = aws_keys['annotation_key'])
    
    # Load publication.pq from ASW S3 instance
    response = s3.get_object(Bucket = aws_keys['bucket_name'], 
                             Key = aws_keys['publication_key'])
    
    content = response['Body'].read()   
    publication_df = pd.read_parquet(BytesIO(content))
    
    # Compare DOI columns //publication vs annotationDOI column//
    annotation_dois = combined_annotations["DOI"]
    publication_dois = publication_df["DOI"]
    dois_to_fetch = annotation_dois[~annotation_dois.isin(publication_dois)].unique().tolist()
    print(f"New total DOIs: {len(dois_to_fetch)}")

    # Fetch data for DOIs that do not exist and append to metadata.pq
    neotomaCrossRef = crossRefQuery(dois_to_fetch)
    neotomaCrossRef = dataPreprocessing(neotomaCrossRef)

    # Update publication_df
    publication_df['subject'] = publication_df['subject'].apply(lambda x: [x.decode('utf-8')]) 
    publication_df = pd.concat([publication_df, neotomaCrossRef], 
                               ignore_index=True)

    # Save the updated publication.pq file
    pq_buffer = BytesIO()
    table = pa.Table.from_pandas(publication_df)
    # Create an empty object in the specified location (key) in the S3 bucket
    s3.put_object(Bucket=aws_keys['bucket_name'], Key='trial_metaupdate.parquet')
    pq.write_table(table, pq_buffer, compression='snappy')
    pq.write_table(table, pq_buffer, compression='snappy')
    #fastparquet.write(pq_buffer, table, compression='SNAPPY')
    pq_buffer.seek(0)
    s3.upload_fileobj(pq_buffer, 
                      Bucket = aws_keys['bucket_name'], 
                      Key = aws_keys['publication_key'])
    

        # Save the updated annotation.pq file
    pq_buffer = BytesIO()
    table = pa.Table.from_pandas(combined_annotations)

    return "Successful Update!"