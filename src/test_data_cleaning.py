"""
Unit testing of data_cleaning.py
"""

import yaml
import pytest
import pandas as pd
from src.data_cleaning import clean_data

config = yaml.safe_load(open("./params.yml"))

@pytest.fixture
def get_clean_data():
    raw_data = pd.read_csv(config['data']['raw_data_path'])
    cleaned_data = clean_data(raw_data)
    return cleaned_data

def test_null(get_clean_data):
    # Check if is nulll is exist
    assert get_clean_data.shape == get_clean_data.dropna().shape

def test_questionmask_exist(get_clean_data):
    # Check if is '?' (default null from system) is exist
    assert "?" not in get_clean_data.values

def test_removed_columns(get_clean_data):
    # Check if removed columns is exist
    removed_columns = ['capital-gain', 'capital-loss']
    for col_name in removed_columns:
        assert col_name not in get_clean_data.columns

def test_race_columns(get_clean_data):
    # Check removed value in 'race' column
    race_value = ["Asian-Pac-Islander", "Amer-Indian-Eskimo"]
    for value in race_value:
        assert value not in list(set(get_clean_data['race']))

def test_education_columns(get_clean_data):
    # Check revmoed value in 'education' column
    education_values = ['11th', '9th', '7th-8th', '5th-6th', 
                        '10th', '1st-4th', 'Preschool', '12th']
    for value in education_values:
        assert value not in list(set(get_clean_data['education']))