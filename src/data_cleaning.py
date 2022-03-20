import argparse
import logging

import pandas as pd

def clean_data(args) -> pd.DataFrame:
    ''' 
    Perform data cleaning task base on insight from
    ./notebook/data_preprocessing.ipynb 
    
    - Remove whitespace in columns name
    - Remove whitespace in string data (all columns)
    - Replace '?' in data and replace it with mode() of each columns
    - Grouping some categorical columns into smaller unique values
    - Remove some column with very high skew distribution

    Arg:
        - args.raw_data_path (str) :
        - args.clean_data_path (str) : 
    Output:
        - data (pd.DataFrame) : 
    '''
    data = pd.read_csv(args.raw_data_path)

    # Remove whitespace from columns name
    data.columns = [col.strip() for col in data.columns]

    # Columns Type Selection
    categorical_columns = data.select_dtypes("object").columns

    # Replace all NA with mode on each columns
    # Remove white space in data
    for cat_col in categorical_columns:
        col_mode = data[cat_col].mode()
        data[cat_col] = data[cat_col].str.strip()
        data[cat_col] = data[cat_col].fillna(str(col_mode))

    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Basic data cleaning")

    parser.add_argument(
        "--raw_data_path", 
        type=str, 
        help="Path of raw data to be cleaned",
        required=True
        )

    parser.add_argument(
        "--clean_data_path", 
        type=str, 
        help="Path of clean data to be cleaned",
        required=True
        )

    args = parser.parse_args()

    clean_data(args)

