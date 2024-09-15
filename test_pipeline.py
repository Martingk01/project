import pytest
import os
import pandas as pd
from pipeline import (
    staging, primary, reporting, main, 
    postcode_cleaning, new_long_col, merging_postcode_UK_crime,
    filling, coord_col, calculate_all_crime_change, crime_per_month,
    regression_coefficient
)  # Import the functions from your pipeline

# Define fixtures
@pytest.fixture(autouse=True)
def cleanup_files():
    """Fixture that runs before each test and ensures any test-generated files are removed after the test."""
    yield  # Run the test
    # Teardown: Remove any files created during the test
    files_to_remove = [
        'staged_Asylum.csv', 'staged_UK_Crime22.csv', 'staged_UK_Crime23.csv', 
        'Merged_data22.csv', 'Merged_data23.csv', 
        'all_region_crime.csv', 'NY_crime_change.csv', 'postcode_dist.csv'
    ]
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)

# Pipeline tests
def test_staging():
    staging()
    assert os.path.exists('staged_Asylum.csv')
    assert os.path.exists('staged_UK_Crime22.csv')
    assert os.path.exists('staged_UK_Crime23.csv')

def test_primary():
    primary()
    assert os.path.exists('Merged_data22.csv')
    assert os.path.exists('Merged_data23.csv')

def test_reporting():
    reporting()
    assert os.path.exists('all_region_crime.csv')
    assert os.path.exists('NY_crime_change.csv')

def test_pipeline_all_stages():
    main('all')
    assert os.path.exists('Merged_data23.csv')
    assert os.path.exists('postcode_dist.csv')

def test_pipeline_staging_only():
    main('staging')
    assert os.path.exists('staged_Asylum.csv')
    assert not os.path.exists('Merged_data23.csv')

def test_pipeline_primary_only():
    main('primary')
    assert os.path.exists('Merged_data22.csv')
    assert not os.path.exists('all_region_crime.csv')

def test_pipeline_reporting_only():
    main('reporting')
    assert os.path.exists('all_region_crime.csv')

def test_invalid_pipeline_stage():
    with pytest.raises(Exception):
        main('invalid_stage')

# Data processing tests
def test_postcode_cleaning():
    data = {
        'Postcode': ['AB1', 'AB2'],
        'District': ['Bristol, City of', 'Manchester'],
        'In Use?': ['Yes', 'No'],
        'Latitude': [51.45, 53.48],
        'Longitude': [-2.58, -2.24],
        'LSOA Code': ['LSOA1', 'LSOA2'],
        'Lower layer super output area': ['Area1', 'Area2'],
        'Ward': ['Ward1', 'Ward2']
    }
    df = pd.DataFrame(data)
    result = postcode_cleaning(df)
    assert len(result) == 1
    assert result.iloc[0]['District'] == 'Bristol'

def test_new_long_col():
    data = {
        'Longitude': [-2.58, -2.24],
        'Latitude': [51.45, 53.48]
    }
    df = pd.DataFrame(data)
    result = new_long_col(df)
    assert 'Coordinates' in result.columns
    assert result['Coordinates'].iloc[0] == (-2.58, 51.45)

def test_merging_postcode_UK_crime():
    df_data = {
        'Longitude': [-2.58, -2.24],
        'Latitude': [51.45, 53.48],
        'Coordinates': [(-2.58, 51.45), (-2.24, 53.48)],
        'Postcode': ['AB1', 'AB2']
    }
    comb_data = {
        'Coord2': ['C1', 'C2'],
        'Longitude': [-2.58, -2.24],
        'Latitude': [51.45, 53.48],
        'Coordinates': [(-2.58, 51.45), (-2.24, 53.48)]
    }
    df = pd.DataFrame(df_data)
    comb = pd.DataFrame(comb_data)
    result = merging_postcode_UK_crime(df, comb)
    assert 'Distance' in result.columns
    assert result['Distance'].iloc[0] == 0

def test_filling():
    data = {
        'LSOA name': ['name 1234', 'no location 1234'],
        'Crime ID': [None, 'B123'],
        'Latitude': [None, 53.48],
        'Longitude': [None, -2.24]
    }
    df = pd.DataFrame(data)
    result = filling(df)
    assert result['Crime ID'].iloc[0] == 'a'
    assert result['Latitude'].iloc[0] == 0
    assert result['LSOA name'].iloc[1] == 'no location 1234'

def test_coord_col():
    data = {
        'Longitude': [-2.58, -2.24],
        'Latitude': [51.45, 53.48]
    }
    df = pd.DataFrame(data)
    result = coord_col(df)
    assert 'Coord2' in result.columns
    assert 'Coordinates' in result.columns
    assert result['Coord2'].iloc[0] == (-2.58, 51.45)
    assert result['Coordinates'].iloc[0] == (-2.58, 51.45)

def test_calculate_all_crime_change():
    data_22 = {
        'Crime ID': [100, 200],
        'Reported by': ['Region1', 'Region2'],
        'Month': ['2022-01', '2022-02']
    }
    data_23 = {
        'Crime ID': [150, 250],
        'Reported by': ['Region1', 'Region2'],
        'Month': ['2023-01', '2023-02']
    }
    df_22 = pd.DataFrame(data_22)
    df_23 = pd.DataFrame(data_23)
    region = ['Region1', 'Region2']
    result = calculate_all_crime_change(df_22, df_23, region)
    assert 'Change in crime' in result.columns
    assert result['Change in crime'].iloc[0] == '50.0%'

def test_crime_per_month():
    data = {
        'Crime ID': ['C1', 'C2'],
        'Reported by': ['Region1', 'Region2'],
        'Month': ['2022-01', '2022-02']
    }
    df = pd.DataFrame(data)
    region = ['Region1']
    result = crime_per_month(df, region)
    assert len(result) == 1
    assert result['Month'].iloc[0] == '01'

def test_regression_coefficient():
    data = {
        'CityName': ['City1', 'City2'],
        'Reported by': ['Region1', 'Region2'],
        'Asylum seekers per 1,000 population': [10, 20],
        'Crimes per 1000 people': [100, 200]
    }
    df = pd.DataFrame(data)
    result = regression_coefficient(df)
    assert result != 0
