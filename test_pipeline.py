import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline import (
    main,
    logging,
    cleaning_asylum,
    remove_code,
    filling,
    coord_col,
    postcode_cleaning,
    new_long_col,
    merging_postcode_UK_crime,
    crimes_per_crime,
    crimes_per_1000,
    calculate_all_crime_change,
    crime_per_month,
    regression_coefficient
)


@pytest.fixture
def df_asylum():
    # Create a DataFrame with empty rows and columns to match the structure
    data = {
        'Unnamed: 0': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 1': [None, None, None, None, 'United Kingdom',None, 'East Midlands', 'East of England', 'London', 'North East', 'North West', None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 2': [None, None, None, 'Contingency accommodation - hotel', 29585,None, 937, 2377, 11226, 353, 3122, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 3': [None, None, None, 'Contingency accommodation - other', 2458,None, 0, 569, 1236, 31, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 4': [None, None, None, 'Dispersed accommodation', 61778,None, 3591, 2119, 4956, 6536, 16088, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 5': [None, None, None, 'Initial accommodation', 1880,None, 209, 0, 626, 0, 396, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 6': [None, None, None, 'Other accommodation', 941,None, 0, 540, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 7': [None, None, None, 'Total', 96642,None, 4737, 5605, 18044, 6920, 19606, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 8': [None, None, None, 'Population', 66980375, None, 4880200, 6334500, 8799800, 2647100, 7417300, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'Unnamed: 9': [None, None, None, 'Asylum seekers per 10,000 population', 14,None, 10, 9, 20.5, 26.1, 26.4, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df


@pytest.fixture
def df_crime():
    data = {
        'Crime ID': [
            None, '5113ac422e61c8150978c61fe11cd909392b690e0883f860a17124b039253322',        #Empty cells to see how my functions deal with null values
            'd14dbe47684d5d51fc62b85ba40b78a4878bc187e25331871c803b454ada3301',
            '5f5332e8f726b16bc717a205ea566cdfde30b2ea5882e852b91c21967bf5e4df', 
            None],
        'Month': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],      
        'Reported by': [
            'Sussex Police', 'Sussex Police', 'North Yorkshire Police',
            'Gloucestershire Constabulary', 'Avon and Somerset Constabulary'],
        'Falls within': [
            'Sussex Police', 'Sussex Police', 'North Yorkshire Police',
            'Gloucestershire Constabulary', 'Avon and Somerset Constabulary'],
        'Longitude': [-0.227676, None, 0.3434, -0.23217, -0.227293],                           #Empty cells to see how my functions deal with null values
        'Latitude': [50.836076, None, 52.3438, 50.843429, 50.84206],
        'Location': [
            'On or near Orchard Close', 'On or near The Crescent', 'On or near Summersdeane', 
            'On or near Downsway', 'On or near Ridgeway Close'],
        'LSOA code': ['E01031349', 'E01031349', 'E01031350', 'E01031350', 'E01031350'],
        'LSOA name': [None, 'Adur 001A', 'Adur 001B', 'Adur 001B', 'Adur City 001B'],           #Empty cells to see how my functions deal with null values
        'Crime type': [
            'Anti-social behaviour', 'Anti-social behaviour', 'Anti-social behaviour', 
            'Criminal damage and arson', 'Violence and sexual offences'],
        'Last outcome category': ['', '', '', 'Status update unavailable', 'Unable to prosecute suspect'],
        'Context': ['', '', '', '', '']
    }
    return pd.DataFrame(data)



@pytest.fixture
def df_postcode():
    return pd.DataFrame({
        'Postcode': ['BS4 4QY', 'BS1 6YH', 'CL9 OT6','CF32 3XY','GH5 4FG'], 
        'In Use?': ['Yes', 'No', 'Yes','Yes','Yes'],                                 #Have a cell with a value of 'no' to see if that row gets removed
        'District': ['Bristol, City of', 'Bristol, City of', 'Something','Arun','Katowice'], #Have a cell with a value Something and Katowice to see if those rows get removed as they dont belong in city dictionary
        'Latitude': [50.836076, 52.2, 52.3436,53.23,55.12],  
        'Longitude': [-0.227676, 0.2, 0.3432,2.3,5.4],
        'LSOA Code':['E153','E245','E142','E612','E415'],
        'Lower layer super output area': ['LSOA 1234', 'LSOA 45b6', 'LSOA 78x9', 'LSOA abcd','LSOA ghfy'],
        'Ward': ['Ward1', 'Ward2', 'Mendip','Ward4','ward5'] #Have a cell with a value Mendip to see if this row doesn't get removed as Mendip exists in Ward dictionary
    })






 #Test cases
def test_cleaning_asylum(df_asylum):
    # Store the length of the 'Population' column before cleaning
    original_population_length = len(df_asylum['Unnamed: 7'])
    
    # Apply the cleaning_asylum function
    result = cleaning_asylum(df_asylum)

    # Store the length of the 'Population' column after cleaning
    result_population_length = len(result['Population'].dropna())

    # Check if the length after cleaning is equal to the original length minus 21 (the number of rows that were removed)
    assert result_population_length == (original_population_length - 21), \
        f"Expected length of 'Population' column to be {original_population_length - 21}, but got {result_population_length}"



def test_remove_code():
    assert remove_code('LSOA 1234') == 'LSOA' #Checking if the function works for 2 elements
    assert remove_code('Bristol City 1234') == 'Bristol City' #checking if the function works for more than 2 elements

def test_filling(df_crime):
    # Apply the filling function
    result = filling(df_crime)

    # Expected DataFrame
    expected_df = pd.DataFrame({
        'Crime ID': [
            'a', '5113ac422e61c8150978c61fe11cd909392b690e0883f860a17124b039253322',
            'd14dbe47684d5d51fc62b85ba40b78a4878bc187e25331871c803b454ada3301',
            '5f5332e8f726b16bc717a205ea566cdfde30b2ea5882e852b91c21967bf5e4df',
            'a'
        ],
        'Month': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],
        'Reported by': [
            'Sussex Police', 'Sussex Police', 'North Yorkshire Police',
            'Gloucestershire Constabulary', 'Avon and Somerset Constabulary'
        ],
        'Crime type': [
            'Anti-social behaviour', 'Anti-social behaviour', 'Anti-social behaviour',
            'Criminal damage and arson', 'Violence and sexual offences'
        ],
        'Longitude': [-0.227676, 0.000000, 0.3434, -0.232170, -0.227293],   
        'Latitude': [50.836076, 0.000000, 52.34389,50.843852, 50.842060],
        'LSOA name': [
            'no location 1234', 'Adur 001A', 'Adur 001B', 'Adur 001B', 'Adur City 001B' #included cities with no location and cities with more than one word to see how filling deals with it 
        ],
        'CityName': [
            'no location', 'Adur', 'Adur', 'Adur', 'Adur City'           
        ]
    })

    # Check if the result DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(result, expected_df)

def test_coord_col(df_crime):
    result = coord_col(df_crime) #applying the test function
    assert 'Coordinates' in result.columns #checking if coordinates column is in the dataframe
    assert result['Coordinates'][0] == (-0.228, 50.836) #checking if the first row in coordinates rounded to 3 decimal places
    assert 'Coord2' in result.columns #checking if coord2 column is in the dataframe
    assert result['Coord2'][0] == (-0.227676, 50.836076)  #checking if the first row in coord is same as coordinates apart it didnt round any decimal places

def test_postcode_cleaning(df_postcode):
    result = postcode_cleaning(df_postcode) #applying the test function
    assert result.iloc[0]['District'] == 'Bristol' #checking if it correctly renamed 'Bristol, City of' to 'Bristol'
    assert len(result) == 3  #the resulting dataframe should have 3 rows due to filtering out specific rows

def test_new_long_col(df_postcode):
    result = new_long_col(df_postcode) #applying the test function
    assert 'Coordinates' in result.columns #checking if the coordinates column exists in the dataframe
    assert result['Coordinates'][0] == (-0.228,50.836) #checking if the first row in coordinates rounded to 3 decimal places

    
def test_merging_postcode_UK_crime(df_postcode, df_crime):
    # Step 1: Test new_long_col function
    try:
        processed_postcode = new_long_col(df_postcode)
    except Exception as e:
        raise AssertionError(f"Error in new_long_col function: {str(e)}")
    
    # Step 2: Test filling function
    try:
        filled_crime = filling(df_crime)
    except Exception as e:
        raise AssertionError(f"Error in filling function: {str(e)}")
    
    # Step 3: Test coord_col function
    try:
        processed_crime = coord_col(filled_crime)
    except Exception as e:
        raise AssertionError(f"Error in coord_col function: {str(e)}")
    
    # Step 4: Test merging_postcode_UK_crime function
    try:
        result = merging_postcode_UK_crime(processed_postcode, processed_crime)
    except Exception as e:
        raise AssertionError(f"Error in merging_postcode_UK_crime function: {str(e)}")

    # Step 5: Check if 'Postcode' column exists in result
    try:
        assert 'Postcode' in result.columns, "'Postcode' column not found in the result DataFrame."
    except AssertionError as e:
        raise AssertionError(f"Postcode column check failed: {str(e)}")

    # Step 6: Check if 'Distance' column values are non-negative
    try:
        assert all(result['Distance'] >= 0), "Not all 'Distance' values are non-negative."
    except Exception as e:
        raise AssertionError(f"Distance check failed: {str(e)}")

@pytest.fixture
def caplog(caplog):
    caplog.set_level(logging.INFO)
    return caplog

# Test the 'staging' pipeline
@patch('pipeline.staging')
def test_staging_pipeline(mock_staging, caplog):
    main('staging')
    
    # Check if staging was called
    mock_staging.assert_called_once()

    # Check logging messages
    assert "Staging execution completed successfully" in caplog.text
    assert "Pipeline run complete (staging)" in caplog.text

# Test the 'primary' pipeline
@patch('pipeline.primary')
def test_primary_pipeline(mock_primary, caplog):
    main('primary')
    
    # Check if primary was called
    mock_primary.assert_called_once()

    # Check logging messages
    assert "Primary execution completed successfully" in caplog.text
    assert "Pipeline run complete (primary)" in caplog.text

# Test the 'reporting' pipeline
@patch('pipeline.reporting')
def test_reporting_pipeline(mock_reporting, caplog):
    main('reporting')
    
    # Check if reporting was called
    mock_reporting.assert_called_once()

    # Check logging messages
    assert "Reporting execution completed successfully" in caplog.text
    assert "Pipeline run complete (reporting)" in caplog.text

# Test the 'all' pipeline (staging -> primary -> reporting)
@patch('pipeline.staging')
@patch('pipeline.primary')
@patch('pipeline.reporting')
def test_all_pipeline(mock_reporting, mock_primary, mock_staging, caplog):
    main('all')
    
    # Check if all stages were called
    mock_staging.assert_called_once()
    mock_primary.assert_called_once()
    mock_reporting.assert_called_once()

    # Check logging messages
    assert "Staging execution completed successfully" in caplog.text
    assert "Primary execution completed successfully" in caplog.text
    assert "Reporting execution completed successfully" in caplog.text

# Test invalid pipeline input
def test_invalid_pipeline_input(caplog):
    main('invalid_stage')

    # Check logging for critical error
    assert "Invalid pipeline stage specified" in caplog.text

# Test exception handling
@patch('pipeline.staging', side_effect=Exception("Staging error"))
def test_pipeline_exception(mock_staging, caplog):
    main('staging')

    # Check if the exception was logged as an error
    assert "Pipeline execution failed: Staging error" in caplog.text

#def test_crimes_per_crime(df_crime):
#    pop_func = lambda city: 1000  # Mock population function
#    result = crimes_per_crime(df_crime, pop_func)
#    assert 'Crimes per 1000 people' in result.columns

#def test_crimes_per_1000(df_crime):
#    pop_func = lambda city: 1000  # Mock population function
 #   result = crimes_per_1000(df_crime, pop_func)
#    assert 'Crimes per 1000 people' in result.columns

#def test_calculate_all_crime_change():
#    all_data22 = pd.DataFrame({'Crime ID': [1, 2, 3], 'Reported by': ['Region A', 'Region B', 'Region A']})
#    all_data23 = pd.DataFrame({'Crime ID': [2, 3, 4], 'Reported by': ['Region A', 'Region B', 'Region A']})
#    result = calculate_all_crime_change(all_data22, all_data23, ['Region A', 'Region B'])
#    assert 'Change in crime' in result.columns

#def test_crime_per_month(df_crime):
#    result = crime_per_month(df_crime, ['Region A'])
#    assert 'Crime ID' in result.columns

#def test_regression_coefficient(df_crime):
#    result = regression_coefficient(coord_col(filling(df_crime)))
#    assert isinstance(result, float)
#    assert result <=1 and result >=-1

