# %%
import pandas as pd 
import glob 
import os
import numpy as np
import logging

# %% [markdown]
# # Data Processing Functions
# 
# This notebook contains various functions for processing and analyzing crime and asylum seeker data. Each function is documented to explain its purpose, steps, and outputs.
# 
# 
# 
# ```python

# %%
'''
This function cleans and processes the asylum seekers data, adjusting for format inconsistencies and calculating asylum seekers per 1,000 population.
'''

def cleaning_asylum(df):
    # 1. Drop the first column named 'Unnamed: 0'
    df = df.drop(columns=['Unnamed: 0'])
    
    # 2. Remove the first 3 rows
    df = df.iloc[3:].reset_index(drop=True)
    
    # 3. Rename the first row as the header
    df.columns = df.iloc[0]  # Set the first remaining row as header
    
    # 4. Drop the row after setting it as header (first row with headers now)
    df = df.drop(df.index[0]).reset_index(drop=True)
    
    # 5. Remove the second row
    df = df.drop(df.index[1]).reset_index(drop=True)
    
    # 6. Remove the bottom 16 rows
    df = df.iloc[:-16].reset_index(drop=True)
    
    # 7. Rename the first column to 'Cities'
    df.rename(columns={df.columns[0]: 'Cities'}, inplace=True)
    
    # 8. Remove any empty rows
    df = df.dropna(how='all').reset_index(drop=True)
    
    # Now the dataframe has empty rows removed, the first column renamed to 'Cities'
    # You can adjust the columns as needed; make sure these column names are correct
    df = df[['Cities', 'Population', 'Asylum seekers per 10,000 population']]

    df['Asylum seekers per 1,000 population'] = df['Asylum seekers per 10,000 population'] *10

    df = df.rename(columns={'Cities':'CityName'})

    return df

'''
This function cleans LSOA names by removing the code part of the name.
'''
def remove_code(x):
    x = str(x)
    if ' ' in x:
        return ' '.join(x.split()[:-1])
    else:
        return x  

    return x 

'''
This function fills missing values in crime data and creates a column for city names based on LSOA names.
'''

def filling(df):
    df['LSOA name'] = df['LSOA name'].fillna('no location 1234')
    df['CityName'] = df['LSOA name'].apply(remove_code)
    df['Crime ID'] = df['Crime ID'].fillna('a')
    df['Latitude'] = df['Latitude'].fillna(0)
    df['Longitude'] = df['Longitude'].fillna(0)

    return df[['Crime ID','Month','Reported by','Crime type','Longitude','Latitude','LSOA name','CityName']]


'''
This function creates new columns for coordinates rounded to 3 decimal places.
'''

def coord_col(df):
    
    df['Coord2'] = list(zip(df['Longitude'], df['Latitude']))
    df['Coordinates'] = list(zip(df['Longitude'].apply(lambda x: round((x),3)), df['Latitude'].apply(lambda x: round((x),3))))
    
    return df

'''
This function cleans and filters postcode data based on its usage and relevance to specific cities or wards.
'''

def postcode_cleaning(df):
    
    df = df[df['In Use?'] == 'Yes']
    df = df[['Postcode','District','Latitude','Longitude','LSOA Code','Lower layer super output area','Ward']]
    df = df[df['District'].isin(cities) | df['Ward'].isin(wrd)]
    # Rename 'Bristol, City of' to 'Bristol' in the 'District' column
    df['District'] = df['District'].replace('Bristol, City of', 'Bristol')
    df = df.reset_index(drop = True)

    return df

'''
This function creates a new column with coordinates rounded to 3 decimal places.
'''

def new_long_col(df):
    # Create a new column 'Longitude_100' by multiplying the 'Longitude' column by 100
    df['Coordinates'] = list(zip(df['Longitude'].apply(lambda x: round((x),3)), df['Latitude'].apply(lambda x: round((x),3))))

    return df

'''
This function merges postcode data with crime data based on coordinates and calculates the distance between them.
'''

def merging_postcode_UK_crime(df,comb):

    df = df[['Longitude','Latitude','Coordinates','Postcode']]
    comb = comb[['Coord2','Longitude','Latitude','Coordinates']]
    
    # Perform the merge on the 'Longitude_100' column in both dataframes
    
    merged_df = comb.merge(df, on='Coordinates', how='left')
    
    
    # The 'how="left"' means it will keep all rows from 'combined' and match with rows from 'df' where possible.
    # If a match doesn't exist in 'df', those rows in 'combined' will have NaN for the 'df' columns.
    merged_df['Distance'] = np.sqrt((merged_df['Longitude_x'] - merged_df['Longitude_y'])**2 + (merged_df['Latitude_x'] - merged_df['Latitude_y'])**2)
    
    
    merged_df = merged_df.dropna(subset=['Coord2', 'Distance'])
    
    grouped_df = merged_df.loc[merged_df.groupby('Coord2')['Distance'].idxmin()]
    
    # Reset index for clean DataFrame output
    grouped_df.reset_index(drop=True, inplace=True)
    
    # Select relevant columns (including the postcode of the smallest distance)
    result_df = grouped_df[['Coord2','Coordinates', 'Distance', 'Postcode']]
    
    return result_df

'''
This function calculates the number of crimes per 1,000 people for specified crime types.
'''

def crimes_per_crime(x, pop_func):
    # Filter out rows where CityName is 'no location'
    b = x[x['CityName'] != 'no location']
    
    # Filter rows where Crime type is 'Burglary', 'Drugs', or 'Violent crime'
    crime_types = ['Burglary', 'Drugs', 'Violence and sexual offences']
    b = b[b['Crime type'].isin(crime_types)]
    
    # Group by CityName and Reported by, and aggregate data
    b = b.groupby(['CityName', 'Reported by']).agg({'Crime type': 'count', 'Latitude': 'mean', 'Longitude': 'mean'}).sort_values(by='Crime type', ascending=False).reset_index()
    
    # Apply the population function and calculate crimes per 1000 people
    b['Population_2023'] = b['CityName'].apply(pop_func).astype(float)
    b['Crimes per 1000 people'] = round(((b['Crime type'] / b['Population_2023']) * 1000), 3)

     # Replace inf values with NaN
    b.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Filter out rows with NaN values in 'Crimes per 1000 people'
    filtered_df = b[b['Crimes per 1000 people'].notna()]
    
    # Apply the condition to filter rows where 'Crimes per 1000 people' is greater than 0
    b = filtered_df[filtered_df['Crimes per 1000 people'] > 1]
    
    # Return the top 8 cities sorted by crimes per 1000 people
    return b.sort_values(by='Crimes per 1000 people', ascending=False).reset_index(drop=True)

'''
This function calculates the number of crimes per 1,000 people, taking asylum seekers into account.
'''

def crimes_per_1000(x,pop_func):
    b = x[x['CityName'] != 'no location']
    b = b.groupby(['CityName', 'Reported by','Asylum seekers per 1,000 population']).agg({'Crime type': 'count', 'Latitude': 'mean', 'Longitude': 'mean'}).sort_values(by='Crime type', ascending=False).reset_index()
    b['Population_2023'] = b['CityName'].apply(pop_func).astype(float)
    b['Crimes per 1000 people'] = round(((b['Crime type'] / b['Population_2023']) * 1000),0)
    b.sort_values( by = 'Crimes per 1000 people', ascending = False).reset_index(drop = True)
    # Replace inf values with NaN
    b.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Filter out rows with NaN values in 'Crimes per 1000 people'
    filtered_df = b[b['Crimes per 1000 people'].notna()]
    
    # Apply the condition to filter rows where 'Crimes per 1000 people' is greater than 0
    filtered_df = filtered_df[filtered_df['Crimes per 1000 people'] > 0]
    
    # Sort and reset index
    result_df = filtered_df.sort_values(by='Crimes per 1000 people', ascending=False).reset_index(drop=True)


    return result_df

'''
This function calculates the percentage change in crime between 2022 and 2023.
'''

def calculate_all_crime_change(all_data22,all_data23,region):

    crime_22 = crime_per_month(all_data22,region)
    crime_23 = crime_per_month(all_data23,region)

    x = crime_22.rename(columns={'Crime ID': 'Crimes_2022'})
    y = crime_23.rename(columns={'Crime ID': 'Crimes_2023'})

    
    combined_df = y.merge(x)

    # Calculate the percentage change in crime between 2022 and 2023
    combined_df['Change in crime 1'] = combined_df.apply(
        lambda x: round(((x['Crimes_2023'] - x['Crimes_2022']) / x['Crimes_2022']) * 100, 2), axis=1
    )

    # Sort the DataFrame by the calculated percentage change
    sorted_df = combined_df.sort_values(by='Change in crime 1', ascending=False)

    # Format the change as a percentage and create a new column
    sorted_df['Change in crime'] = sorted_df['Change in crime 1'].apply(lambda x: f'{x}%')

    # Return the final sorted DataFrame with the relevant columns
    return sorted_df.reset_index(drop=True)

'''
This function counts the number of crimes per month for a specific region.
'''

def crime_per_month(y,region):
    y = y[y['Reported by'].isin(region)]

    #y = y[y['Reported by'] == region]
    a = y.groupby(['Month'])[['Crime ID']].count().reset_index()
    y = str(y)
    a['Month'] = a['Month'].apply(lambda x: ''.join(x.split('-')[-1]))
    a['Month'] = a['Month'].apply(month)

    return a 

'''
This function calculates the correlation between asylum seekers per 1,000 population and crimes per 1,000 people.
'''

def regression_coefficient(x):
    x = crimes_per_1000(x,City_pop_a23)[['CityName','Reported by','Asylum seekers per 1,000 population','Crimes per 1000 people']]
    
    x['Asylum seekers per 1,000 population'] = x['Asylum seekers per 1,000 population'].astype(float).round().astype(int)
    
    correlation = x['Asylum seekers per 1,000 population'].corr(x['Crimes per 1000 people'])
    
    
    
    return (correlation)

# %%
'''Created dictionaries for each city in each region and a month dictionary
'''



def City_pop_a23(x):
    pop_dict = { 'Bristol' : '483000',
                 'South Gloucestershire' : '290400',
                 'North Somerset' : '219544',
                 'Bath and North East Somerset' : '196739',
                 'South Somerset' : '173646',
                 'Sedgemoor' : '126679',
                 'Taunton Deane' : '124300',
                 'Mendip' : '118492',
                 'Somerset West and Taunton' : '159640',
                 'York' : '202821',
                 'Harrogate' : '164100',
                 'Scarborough' : '109000',
                 'Selby' : '92400',
                 'Hambleton' : '92000',
                 'Craven' : '57500',
                 'Ryedale' : '56000',
                 'Richmondshire' : '43000',
                 'Brighton and Hove' : '281000',
                 'Crawley' : '118600',
                 'Arun' : '170000',
                 'Eastbourne' : '101600',
                 'Worthing' : '112240',
                 'Hastings' : '91000',
                 'Chichester' : '31000',
                 'Mid Sussex' : '154000',
                 'Horsham' : '50223',
                 'Gloucester' : '133522',
                 'Cheltenham' : '121000',
                 'Stroud' : '120000',
                 'Tewkesbury' : '95000',
                 'Forest of Dean' : '87000',
                 'Cotswold' : '90000'
                 
               }
    return pop_dict.get(x, '0')

def month(x):
    mon_dict = { '01' : 'January',
                 '02' : 'February',
                 '03' : 'March',
                 '04' : 'April',
                 '05' : 'May',
                 '06' : 'June',
                 '07' : 'July',
                 '08' : 'August',
                 '09' : 'September',
                 '10' : 'October',
                 '11' : 'November',
                 '12' : 'December'
               }
    return mon_dict[x]

cities = ['Bristol, City of', 'South Gloucestershire', 'North Somerset', 'Bath and North East Somerset', 
          'South Somerset', 'Sedgemoor', 'Taunton Deane', 'Mendip', 'Somerset West and Taunton',
          
          'York', 'Harrogate', 'Scarborough', 'Selby', 'Hambleton', 'Craven', 'Ryedale', 'Richmondshire',
          
          'Brighton and Hove', 'Crawley', 'Arun', 'Eastbourne', 'Worthing', 'Hastings', 'Chichester', 'Mid Sussex', 'Horsham',
          
          'Gloucester', 'Cheltenham', 'Stroud', 'Tewkesbury', 'Forest of Dean', 'Cotswold']

wrd = ['Mendip', 'Craven', 'North Richmondshire']


month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']

# %% [markdown]
# # Importing and Combining Crime Data
# 
# This notebook demonstrates how to import and combine crime data from multiple police forces for specific years. Each section includes a function to import CSV files and combine them into a single DataFrame for analysis.
# 
# 
# 

# %%
'''Importing dataset for a specific region for a specific year
'''


''' Avon and Sommerset'''
# Function to import all CSV files for a specific year and police force
def import_csv_files(year):
    files = glob.glob(f'Police data/{year}-*/{year}-*-avon-and-somerset-street.csv')
    dataframes = [pd.read_csv(f) for f in files]
    return dataframes

# Importing Avon Somerset crime data for the year 2023
a_23 = import_csv_files(2023)

# Importing Avon Somerset crime data for the year 2022
a_22 = import_csv_files(2022)

# concanating dataframes into one dataframe for specific year
a_23 = pd.concat(a_23, ignore_index=True)
a_22 = pd.concat(a_22, ignore_index=True)

''' Gloucestershire'''
# Function to import all CSV files for a specific year and police force
def import_csv_files(year):
    files = glob.glob(f'Police data/{year}-*/{year}-*-gloucestershire-street.csv')
    dataframes = [pd.read_csv(f) for f in files]
    return dataframes

# Importing gloucestershire crime data for the year 2023
g_23 = import_csv_files(2023)

# Importing gloucestershire street crime data for the year 2022
g_22 = import_csv_files(2022)

# concanating dataframes into one dataframe for specific year
g_23 = pd.concat(g_23, ignore_index=True)
g_22 = pd.concat(g_22, ignore_index=True)

'''North Yorkshire'''
# Function to import all CSV files for a specific year and police force
def import_csv_files(year):
    files = glob.glob(f'Police data/{year}-*/{year}-*-north-yorkshire-street.csv')
    dataframes = [pd.read_csv(f) for f in files]
    return dataframes

# Importing Avon Somerset crime data for the year 2023
ny_23 = import_csv_files(2023)

# Importing Avon Somerset crime data for the year 2022
ny_22 = import_csv_files(2022)

# concanating dataframes into one dataframe for specific year
ny_23 = pd.concat(ny_23, ignore_index=True)
ny_22 = pd.concat(ny_22, ignore_index=True)

'''Sussex'''
# Function to import all CSV files for a specific year and police force
def import_csv_files(year):
    files = glob.glob(f'Police data/{year}-*/{year}-*-sussex-street.csv')
    dataframes = [pd.read_csv(f) for f in files]
    return dataframes

# Importing Avon Somerset crime data for the year 2023
s_23 = import_csv_files(2023)

# Importing Avon Somerset crime data for the year 2022
s_22 = import_csv_files(2022)

# concanating dataframes into one dataframe for specific year
s_23 = pd.concat(s_23, ignore_index=True)
s_22 = pd.concat(s_22, ignore_index=True)    

combined_22 = pd.concat([a_22, g_22,ny_22, s_22], ignore_index=True)
combined_23 = pd.concat([a_23, g_23,ny_23, s_23], ignore_index=True)

# %%
# Define the local data path where all files are stored
LOCAL_DATA_PATH = './'

# File path for the raw asylum seekers data in Excel format
RAW_Asylum = os.path.join(LOCAL_DATA_PATH, 'Asylum Seekers.xlsx')

# DataFrames for UK crime data for the years 2022 and 2023
# combined_22 and combined_23 are assumed to be preloaded DataFrames containing crime data
RAW_UK_Crime22 = combined_22  # DataFrame for UK crime data in 2022
RAW_UK_Crime23 = combined_23  # DataFrame for UK crime data in 2023

# File paths for staged data that will be processed or used later
STAGED_Asylum = os.path.join(LOCAL_DATA_PATH, 'staged_Asylum.csv')  # Staged asylum seekers data
STAGED_UK_Crime22 = os.path.join(LOCAL_DATA_PATH, 'staged_UK_Crime22.csv')  # Staged UK crime data for 2022
STAGED_UK_Crime23 = os.path.join(LOCAL_DATA_PATH, 'staged_UK_Crime23.csv')  # Staged UK crime data for 2023

# File paths for postcode data
RAW_postcode = os.path.join(LOCAL_DATA_PATH, 'postcodes.csv')  # Raw postcode data in CSV format
STAGED_postcode = os.path.join(LOCAL_DATA_PATH, 'staged_postcode.csv')  # Staged postcode data

# File paths for merged data files
MERGED_data22 = os.path.join(LOCAL_DATA_PATH, 'Merged_data22.csv')  # Merged data for 2022
MERGED_data23 = os.path.join(LOCAL_DATA_PATH, 'Merged_data23.csv')  # Merged data for 2023

# File paths for crime data analysis files
all_region_crime1 = os.path.join(LOCAL_DATA_PATH, 'all_region_crime.csv')  # Crime data for all regions
NY_crime_change1 = os.path.join(LOCAL_DATA_PATH, 'NY_crime_change.csv')  # Crime change data for North Yorkshire
G_crime_change1 = os.path.join(LOCAL_DATA_PATH, 'G_crime_change.csv')  # Crime change data for Gloucestershire

# File paths for crime distribution and statistics
crime_dist_231 = os.path.join(LOCAL_DATA_PATH, 'crime_dist_23.csv')  # Crime distribution data for 2023
crimes_1000_231 = os.path.join(LOCAL_DATA_PATH, 'crimes_1000_23.csv')  # Crimes per 1000 people for 2023

# File path for postcode distribution data
postcode_dist1 = os.path.join(LOCAL_DATA_PATH, 'postcode_dist.csv')  # Postcode distribution data


# %%
# Configure logging
logging.basicConfig(
    filename='pipeline.log',  # Log file name
    filemode='a',             # Append mode
    format='%(asctime)s %(levelname)s: %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    level=logging.INFO  # Log level
)


# %%
def staging():
    logging.info("Starting Staging Layer")  # Log that the staging process is starting
    
    try:
        # Fill empty cells in columns 'Crime ID' and 'LSOA name' in the combined UK crime data for 2022 and 2023
        UK_crime22 = filling(RAW_UK_Crime22)  # Apply the filling function to the 2022 UK crime data
        UK_crime23 = filling(RAW_UK_Crime23)  # Apply the filling function to the 2023 UK crime data

        logging.info("Crime data for 2022 and 2023 filled successfully.")  # Log successful completion of crime data filling

        # Load and clean the postcode dataset
        postcode = pd.read_csv(RAW_postcode)  # Read the postcode data from CSV file
        postcode = postcode_cleaning(postcode)  # Apply cleaning to the postcode data
        logging.info("Postcode data cleaned successfully.")  # Log successful completion of postcode data cleaning

        # Load the raw Asylum Seekers data from Excel
        df = pd.read_excel(RAW_Asylum)  # Read the asylum seekers data from Excel file
        logging.info("Asylum seekers data loaded successfully.")  # Log successful loading of asylum seekers data

        # Clean the Asylum Seekers data
        df = cleaning_asylum(df)  # Apply cleaning function to the asylum seekers data
        logging.info("Asylum seekers data cleaned successfully.")  # Log successful cleaning of asylum seekers data

        # Save the cleaned datasets to CSV files
        df.to_csv(STAGED_Asylum, index=False)  # Save the cleaned asylum seekers data to CSV
        UK_crime22.to_csv(STAGED_UK_Crime22, index=False)  # Save the cleaned UK crime data for 2022 to CSV
        UK_crime23.to_csv(STAGED_UK_Crime23, index=False)  # Save the cleaned UK crime data for 2023 to CSV
        postcode.to_csv(STAGED_postcode, index=False)  # Save the cleaned postcode data to CSV
        
        logging.info("Staging data saved successfully.")  # Log successful saving of all staged data
    except Exception as e:
        # Log any exceptions that occur during the staging process
        logging.error(f"Error in staging layer: {e}")  # Log the error message
        raise  # Re-raise the exception to propagate the error


# %%
def primary():
    logging.info("Starting Primary Layer")
    try:
        # Load staging data
        asylum = pd.read_csv(STAGED_Asylum)
        UK_crime22 = pd.read_csv(STAGED_UK_Crime22)
        UK_crime23 = pd.read_csv(STAGED_UK_Crime23)
        postcode = pd.read_csv(STAGED_postcode)
        
        logging.info("Staging data loaded successfully.")
        
        # Apply transformations
        UK_crime22 = coord_col(UK_crime22) 
        UK_crime23 = coord_col(UK_crime23)
        postcode = new_long_col(postcode)
        
        logging.info("Transformations applied to crime and postcode data.")

        # Merge datasets
        crime_postcode_merge22 = merging_postcode_UK_crime(postcode, UK_crime22)
        crime_postcode_merge23 = merging_postcode_UK_crime(postcode, UK_crime23)

        crime_postcode_merge22 = UK_crime22.merge(crime_postcode_merge22, on='Coord2', how='left')
        crime_postcode_merge23 = UK_crime23.merge(crime_postcode_merge23, on='Coord2', how='left')

        crime_postcode_merge22 = crime_postcode_merge22.merge(asylum, on='CityName', how='left')
        crime_postcode_merge23 = crime_postcode_merge23.merge(asylum, on='CityName', how='left')

        logging.info("Datasets merged successfully.")
        
        # Save merged data
        crime_postcode_merge22.to_csv(MERGED_data22, index=False)
        crime_postcode_merge23.to_csv(MERGED_data23, index=False)
        
        logging.info("Primary layer data saved successfully.")
    except Exception as e:
        logging.error(f"Error in Primary Layer: {e}")
        raise


# %%
def reporting():
    logging.info("Starting Reporting Layer")
    try:
        # Load merged data
        all_data22 = pd.read_csv(MERGED_data22)
        all_data23 = pd.read_csv(MERGED_data23)
        all_data = pd.concat([all_data22, all_data23]).reset_index(drop=True)

        logging.info("Merged data loaded successfully.")

        # Calculate crime changes and other statistics
        all_region_crime = calculate_all_crime_change(all_data22, all_data23, ['New Yorkshire Police', 'Avon and Somerset Constabulary', 'Gloucestershire Constabulary', 'Sussex Police'])
        NY_crime_change = calculate_all_crime_change(all_data22, all_data23, ['New Yorkshire Police'])
        G_crime_change = calculate_all_crime_change(all_data22, all_data23, ['Gloucestershire Constabulary'])

        logging.info("Crime change calculations completed.")

        crime_dist_23 = crimes_per_crime(all_data23, City_pop_a23)
        crimes_1000_23 = crimes_per_1000(all_data23, City_pop_a23)

        logging.info("Crime distribution and per-1000 stats calculated.")

        # Group by postcode
        postcode_dist = all_data.groupby(['Postcode']).agg({'Crime type': 'count'})
        postcode_dist = postcode_dist.iloc[1:]

        # Calculate correlation
        correlation_crime_asylum = regression_coefficient(all_data23)

        logging.info("Postcode distribution and correlation analysis completed.")

        # Save to CSV
        all_region_crime.to_csv(all_region_crime1, index=False)
        NY_crime_change.to_csv(NY_crime_change1, index=False)
        G_crime_change.to_csv(G_crime_change1, index=False)
        crime_dist_23.to_csv(crime_dist_231, index=False)
        crimes_1000_23.to_csv(crimes_1000_231, index=False)
        postcode_dist.to_csv(postcode_dist1, index=False)

        logging.info("Reporting data saved successfully.")
        return 'Reporting process completed successfully.'
    except Exception as e:
        logging.error(f"Error in Reporting Layer: {e}")
        raise


# %%
def main(pipeline='all'):
    logging.info("Pipeline execution started")
    try:
        # Run the 'staging' step if requested or if running the entire pipeline
        if pipeline in ['all', 'staging']:
            staging()
            logging.info("Staging execution completed successfully")
            if pipeline == 'staging':
                logging.info("Pipeline run complete (staging)")
                return
        
        # Run the 'primary' step if requested or if running the entire pipeline
        if pipeline in ['all', 'primary']:
            primary()
            logging.info("Primary execution completed successfully")
            if pipeline == 'primary':
                logging.info("Pipeline run complete (primary)")
                return
        
        # Run the 'reporting' step if requested or if running the entire pipeline
        if pipeline in ['all', 'reporting']:
            reporting()
            logging.info("Reporting execution completed successfully")
            if pipeline == 'reporting':
                logging.info("Pipeline run complete (reporting)")
                return
        
        # Handle invalid inputs
        if pipeline not in ['all', 'staging', 'primary', 'reporting']:
            logging.critical("Invalid pipeline stage specified. Please choose 'staging', 'primary', 'reporting', or 'all'.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")

# Example entry point execution
main('all')  # Replace 'reporting' with 'all', 'primary', or 'staging' as needed


# %%



