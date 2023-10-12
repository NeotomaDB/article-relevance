import argparse
import article_relevance as ar

###
# Script to be run from the terminal. 
# Updates AWS S3 instance following objects:
# Annotations.parquet & PublicationMetadata.parquet 

def main():
    parser = argparse.ArgumentParser(description="Pass in a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file")

    args = parser.parse_args()
    csv_file = args.csv_file

    try:
        data = ar.updateSource(csv_file)
        return 
        
    except FileNotFoundError:
        print(f"File '{csv_file}' not found.")
        return

if __name__ == "__main__":
    main()
