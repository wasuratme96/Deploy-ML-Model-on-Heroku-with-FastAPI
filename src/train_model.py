# Script to train machine learning model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import fbeta_score, precision_score, recall_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Args:
        - X_train (np.array) : Training data.
        - y_train (np.array) : 
        - Labels () :
    Outputs:
        None
    -------
    model
        Trained machine learning model.
    """

    pass


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Args:
        - y (np.array) : Known labels, binarized.
        - preds (np.array) : Predicted labels, binarized.
    Ouputs:
        - precision (float) :
        - recall (float) :
        - fbeta (float) :
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.
    Args:
        - model (???): Trained machine learning model.
        - X (np.array) : Data used for prediction.
    Outputs:
        - preds (np.array) :Predictions from the model.
    """
    pass

def execute_modeling(args_data, args_model):
    '''
    Train and save machine learning model
    Args:
        - args_data.featureized_data_path (str) : 
        - args_data.label_data_path (str) :
        - args_mdoel.train_test_split (float) :
        - 
    Outputs:
        None
    '''
    feature_data = np.load(args_data['featurized_data_path'])
    label_data = np.load(args_data['label_data_path'])


    

