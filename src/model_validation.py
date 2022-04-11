import logging
import pandas as pd
import numpy as np
from numpy import mean, std

from joblib import load
from typing import Tuple, List, Dict

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score

from src.preprocess_data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def compute_model_metrics(y:np.array, preds:np.array) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Args:
        - y (np.array) : Known labels, binarized.
        - preds (np.array) : Predicted labels, binarized.
    Ouputs:
        - precision (float) : Precision score
        - recall (float) : Recall score
        - fbeta (float) : F1-Beta score
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def make_prediction(model, X:np.array) -> np.array:
    """ Run model inferences and return the predictions.
    Args:
        - model (GradientBoostingClassifier): Trained machine learning model.
        - X (np.array) : Data used for prediction.
    Outputs:
        - preds (np.array) :Predictions from the model.
    """
    y_preds = model.predict(X)
    return y_preds

def model_slicing_score(test_data: pd.DataFrame,
                        encoder_model: OneHotEncoder,
                        label_bin_model: LabelBinarizer, 
                        trained_model,
                        args_process_data: Dict) -> List[str]:
    """
    Args:
        - test_data (pd.DataFrame) :
        - encoder_model (OneHotEncoder) :
        - label_bin_model (LabelBinarizer) :
        - trained_model (???) :
        - args_process_data (Dict) : 
    Outputs:
        - slices_performance_scoring (List[str]) :
    """
    slices_performance_scoring = []
    logging.info("[Model Validation wih Slicing Data] : Starting validate on slice data")
    for cat_col in args_process_data['cat_features']:
        for class_in_cat in list(set(test_data[cat_col])):
            test_data_slice = test_data[test_data[cat_col] == class_in_cat]
            X_slice, y_slice, _, _ = process_data(test_data_slice,
                                                  categorical_features=args_process_data['cat_features'],
                                                  label=args_process_data['label'],
                                                  encoder=encoder_model,
                                                  lb=label_bin_model,
                                                  training=args_process_data['interfence_mode']['train'])

            y_preds_slice = trained_model.predict(X_slice)
            prc, rcl, fb = compute_model_metrics(y_slice, y_preds_slice)

            line = "[Features : %s -> Value : %s], Precision: %.3f " \
                   "Recall: %.3f FBeta: %.3f" % (cat_col, class_in_cat, prc, rcl, fb)
            logging.info(line)
            slices_performance_scoring.append(line)

    return slices_performance_scoring

def model_test_score(test_data: pd.DataFrame,
                    encoder_model: OneHotEncoder,
                    label_bin_model: LabelBinarizer, 
                    trained_model,
                    args_process_data: Dict,
                    args_model: Dict) -> pd.DataFrame:
        '''
         Args:
            - test_data (pd.DataFrame) :
            - encoder_model (OneHotEncoder) :
            - label_bin_model (LabelBinarizer) :
            - trained_model (???) :
            - args_process_data (Dict) : 
        Outputs:
            None
        '''
        X_test, y_test, _, _ = process_data(test_data,
                                                  categorical_features=args_process_data['cat_features'],
                                                  label=args_process_data['label'],
                                                  encoder=encoder_model,
                                                  lb=label_bin_model,
                                                  training=args_process_data['interfence_mode']['train'])

        cv = KFold(n_splits=args_model['cv_splits'], shuffle = True, random_state=args_model['random_state'])
        test_scores = cross_val_score(trained_model, X_test, y_test, 
                            scoring = "accuracy", cv = cv, n_jobs=-1)

        y_test_preds = trained_model.predict(X_test)
        prc_test, rcl_test, fb_test = compute_model_metrics(y_test, y_test_preds)

        logger.info("[Model Evaluation Steps] : Accuracy = %.3f (mean) %.3f (std)" % (mean(test_scores), std(test_scores)))
        logger.info("[Model Evaluation Steps] : Precision: %.3f Recall: %.3f FBeta: %.3f" % (prc_test, rcl_test, fb_test))
        return None

def execute_model_validation(args_test_data, args_process_data, args_model):
    """
    Args:
        - args_test_data 
        - args_process_data
    Outputs:
        None
    """
    logger.info("[Model Evaluation Steps] : Start evaluate model")

    test_data = pd.read_csv(args_test_data['full_data_path'])
    encoder = load(args_process_data['interfence_mode']['encoder'])
    lb = load(args_process_data['interfence_mode']['label_bin'])
    model = load(args_model['model_path'])

    _ = model_test_score(test_data, encoder, lb,
                        model, args_process_data, args_model)

    slice_scoring_list = model_slicing_score(test_data, encoder, lb, 
                                             model, args_process_data)

    with open(args_model['model_scoring_path'], 'w') as out:
        for slice_value in slice_scoring_list:
            out.write(slice_value + '\n')