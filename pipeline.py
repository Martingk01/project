import os
import logging
import pandas as pd
import numpy as np

# Constants
LOCAL_DATA_PATH = './'
LOG_FILE = os.path.join(LOCAL_DATA_PATH, 'pipeline.log')
RAW_STREET = os.path.join(LOCAL_DATA_PATH, '2022-01-cheshire-street.csv')
RAW_OUTCOMES = os.path.join(LOCAL_DATA_PATH, '2022-01-cheshire-outcomes.csv')
STAGED_STREET = os.path.join(LOCAL_DATA_PATH, 'staged_cheshire_street.csv')
STAGED_OUTCOMES = os.path.join(LOCAL_DATA_PATH, 'staged_cheshire_outcomes.csv')
PRIMARY_DATA_FILE = os.path.join(LOCAL_DATA_PATH, 'primary_cheshire.csv')
REPORTING_DATA_FILE = os.path.join(LOCAL_DATA_PATH, 'reporting_cheshire.csv')


# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def ingest_data(file_path: str)-> pd.DataFrame:
    """
    Ingest raw data from a CSV file. Pass in the file path as a string and returns a pandas dataframe.
    """
    logging.info(f"Starting data ingestion from {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data ingestion from {file_path} completed successfully")
        return df
    except Exception as e:
        logging.error(f"Error reading the CSV file {file_path}: {e}")
        raise ValueError(f"Error reading the CSV file {file_path}: {e}")
    
def split_month(df):
    """
    Separate the field 'Month' into 'Date_Year' and 'Date_Month'
    """
    df = df.join(
    df['Month'].str.split('-', n=2, expand=True).rename(columns={0:'Date_Year', 1:'Date_Month'})
                )
    return df

def merge_data(df, df_outcomes):
    """
    Merge the main data with outcomes data on 'Crime ID'.
    """
    return pd.merge(df, df_outcomes[['Crime ID', 'Outcome type']], how='left', on='Crime ID')

def finaloutcome(df):
    """
    Create 'Final Outcome' column based on 'Outcome type' and 'Last outcome category'.
    """
    df['Final Outcome'] = np.where(
        df['Outcome type'].notnull(), df['Outcome type'], df['Last outcome category']
        )
    return df

def categorize_outcome(outcome):
    if outcome in ['Unable to prosecute suspect', 
                   'Investigation complete; no suspect identified', 
                   'Status update unavailable']:
        return 'No Further Action'
    elif outcome in ['Local resolution', 
                     'Offender given a caution', 
                     'Action to be taken by another organisation']:
        return 'Non-criminal Outcome'
    elif outcome in ['Further investigation is not in the public interest', 
                     'Further action is not in the public interest', 
                     'Formal action is not in the public interest']:
        return 'Public Interest Consideration'
    else:
        return 'Unknown'  # Or any other category for unknown outcomes

def apply_categorization(df):
    """
    Apply categorization to 'Final Outcome' column.
    """
    df['Broad Outcome Category'] = df['Final Outcome'].apply(categorize_outcome)
    return df

def del_cols(df: pd.DataFrame, list_of_cols:list):
    """
    Delete unnecessary columns from the DataFrame.
    """
    df.drop(columns=list_of_cols, inplace=True)
    return df

def staging():
    """
    Ingest the data, apply cleaning, and store to CSV files for staging.
    """
    logging.info("Starting Staging Layer")
     # ingest raw
    df = ingest_data(RAW_STREET)
    df_outcomes = ingest_data(RAW_OUTCOMES)
    try:
        # Apply transformations
        df = split_month(df)
        df_outcomes = split_month(df_outcomes)
        df = del_cols(df, ['Month', 'Context'])
        df_outcomes = del_cols(df_outcomes, ['Month'])
        # Save staging files to CSV
        df.to_csv(STAGED_STREET, index=False)#
        df_outcomes.to_csv(STAGED_OUTCOMES, index=False)
        logging.info("Data staging completed successfully")
    except Exception as e:
        logging.error(f"Error during data staging: {e}")

def primary():
    """
    Primary Layer: Store the transformed data to a CSV file.
    """
    logging.info("Starting Primary Layer")
     # ingest staging
    df = ingest_data(STAGED_STREET)
    df_outcomes = ingest_data(STAGED_OUTCOMES)
    try:
        # merge 
        mdf = merge_data(df, df_outcomes)
        # Apply primary transformations
        mdf = finaloutcome(mdf)
        mdf = apply_categorization(mdf)
        # Save to CSV
        mdf.to_csv(PRIMARY_DATA_FILE, index=False)
        logging.info("Primary Layer completed successfully")
    except Exception as e:
        logging.error(f"Error during Primary Layer: {e}")

def reporting():
    """
    Reporting Layer: Store the aggregated reporting data to a CSV file.
    """
    logging.info("Starting Reporting Layer.")
    rdf = ingest_data(PRIMARY_DATA_FILE)
    try:
        # Apply aggregation directly within the function
        agg_df = rdf.groupby(['Crime type', 'Broad Outcome Category']).size().reset_index(name='Count')

        # Save to CSV
        agg_df.to_csv(REPORTING_DATA_FILE, index=False)
        logging.info("Reporting data aggregation completed successfully")
    except Exception as e:
        logging.error(f"Error during reporting data aggregation: {e}")

def main(pipeline='all'):
    logging.info("Pipeline execution started")

    try:
        if pipeline in ['all', 'staging', 'primary', 'reporting']:
            staging()
            logging.info("Staging execution completed successfully")
            if pipeline == 'staging':
                # If only staging is requested, print success and return
                logging.info("Pipeline run complete")
                return
            # Process the staged data
            primary()
            logging.info("Primary execution completed successfully")
            if pipeline == 'primary':
                # If only primary is requested, print success and return 
                logging.info("Pipeline run complete")
                return
            # Generate reports based on processed data
            reporting()
            logging.info("Reporting execution completed successfully")
            if pipeline == 'reporting':
                logging.info("Pipeline run complete")
                return
            logging.info("Full pipeline run complete")
        else:
            # Inform the user about an invalid pipeline stage input
            logging.critical("Invalid pipeline stage specified. Please choose 'staging', 'primary', 'reporting', or 'all'.")
    except Exception as e:
        # Catch and print any exceptions occurred during pipeline execution
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
    
