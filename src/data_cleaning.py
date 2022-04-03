import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def clean_data(data:pd.DataFrame) -> None:
    ''' 
    Perform data cleaning task base on insight from
    ./notebook/data_preprocessing.ipynb 
    
    - Remove whitespace in columns name
    - Remove whitespace in string data (all columns)
    - Replace '?' in data and replace it with mode() of each columns
    - Grouping some categorical columns into smaller unique values
    - Remove some column with very high skew distribution

    Arg:
        - data (pd.DataFrame) :
    Output:
        None
    '''
    # Remove whitespace from columns name
    logger.info("[Clean Data Steps] : Remove whitespace in columns name")
    data.columns = [col.strip() for col in data.columns]

    # Columns Type Selection
    categorical_columns = data.select_dtypes("object").columns

    # Replace '?' with np.nan to make replace with imputed value
    logger.info("[Clean Data Steps] : Dealing with missing value")
    data.replace('?', np.nan, inplace = True)
    data.replace('? ', np.nan, inplace = True)
    data.replace(' ?', np.nan, inplace = True)

    # Replace all NA with mode on each columns
    # Remove white space in data
    for cat_col in categorical_columns:
        col_mode = data[cat_col].mode()
        data[cat_col] = data[cat_col].str.strip()
        data[cat_col] = data[cat_col].fillna(str(col_mode))

    # Group naitve-country into "United-State" and "Other"
    logger.info("[Clean Data Steps] : Grouping categorical data")
    data['native-country'] = np.where(data['native-country'] != 'United-States', 
                                                        'Other', 'United-States')

    # Group race into "White", "Black" and "Other"
    data['race'] = np.where((data['race'] == 'Asian-Pac-Islander') |
                            (data['race'] == 'Amer-Indian-Eskimo'), 'Other', data['race'])

    data['education'].replace(['11th', '9th', '7th-8th', '5th-6th', 
                               '10th', '1st-4th', 'Preschool', '12th'], 'School', inplace = True)
    
    # Drop very skew data on numeric columns
    logger.info("[Clean Data Steps] : Drop skew numerical data")
    data.drop(columns = ['capital-gain', 'capital-loss'], inplace = True)         
    
    return data

def execute_clean_data(args):
    ''' 
    Arg:
        - args.raw_data_path (str) :
        - args.clean_data_path (str) : 
    Output:
        None
    '''
    # Dowload Raw Data
    logger.info("[Clean Data Steps] : Dowload raw data")
    data = pd.read_csv(args['raw_data_path'])
    
    # Use clean_data function
    data_clean = clean_data(data)

    # Save Clean Data
    logger.info("[Clean Data Steps] : Save clean data")
    data_clean.to_csv(args['clean_data_path'], index = 0)

