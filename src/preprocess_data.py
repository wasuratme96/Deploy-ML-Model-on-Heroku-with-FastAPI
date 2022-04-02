import logging
import numpy as np
import pandas as pd

from typing import *
from joblib import dump
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def process_data(
    X : pd.DataFrame, 
    categorical_features : list = [], 
    label: str = None, 
    training:bool = True, 
    encoder: OneHotEncoder = None, 
    lb: LabelBinarizer = None
    ):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    logger.info("[Feature Engineering Steps] : Seperate Label - Features")
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    logger.info("[Feature Engineering Steps] : Encoding & Labelize on feature and label")
    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()

        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def execute_data_split(args_data:str, args_model:float) -> None:
    '''
    Args:
        - args_data.clean_data_path (str):
        - args_data.train.full_data_path (str):
        - args_data.test.full_data_path (str):
        - args_model.tes_ratio (float)
    Outputs:
        None
    '''
    logger.info("[Data Splitting Steps] : Download Clean Data")
    clean_data = pd.read_csv(args_data['clean_data_path'])

    logger.info("[Data Splitting Steps] : Split Train-Test Data")
    train_data, test_data = train_test_split(clean_data, 
                                            test_size=args_model['test_ratio'],
                                            random_state = args_model['random_state'])

    logger.info("[Data Splitting Steps] : Save Train-Test Data")
    train_data.to_csv(args_data['train']['full_data_path'], index = 0)
    test_data.to_csv(args_data['test']['full_data_path'], index = 0)
    
def execute_process_data(args_data: str, 
                        args_feature: str, 
                        args_feature_mode: str,
                        args_model: str):
    '''
    Args:
        - args_data.clean_data_path (str) : Path of clean data in .csv format
        - args_data.featurized_data_path (str) : Path to store featurized data in .csv format
        - args_feature.cat_features (list[str]) : List of categorical data
        - args_feature.label (str) : Columns name of label data
        - args_feature.train_mode (bool) : if True meaning is training mode, False is inference mode
        - args_feature.encoder (OneHotEncoder) : OneHotEncoder object in .joblib format
        - args_feature.label_encoder (LabelBinarizer) : LabelBinarizer object in .joblib format
    Outputs:
        None
    '''
    train_data = pd.read_csv(args_data['train']['full_data_path'])
    feat_data, label_data, encoder, lb = process_data(train_data,
                                                      args_feature['cat_features'],
                                                      args_feature['label'],
                                                      args_feature_mode['train'],
                                                      args_feature_mode['encoder'],
                                                      args_feature_mode['label_bin'])

    np.save(args_data['train']['featurized_data_path'], feat_data)
    np.save(args_data['train']['label_data_path'], label_data)
    dump(encoder, args_feature['interfence_mode']['encoder'])
    dump(lb, args_feature['interfence_mode']['label_bin'])