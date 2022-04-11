# Script to train machine learning model.
import logging
import numpy as np
from numpy import mean, std
from joblib import dump

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from src.model_validation import compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def train_model(X_train: np.array, 
                y_train: np.array, 
                args_model:dict) -> GradientBoostingClassifier:
    """
    Trains a machine learning model GradientBoostingClassifier
    and returns trained model.
    Args:
        - X_train (np.array) : Training data.
        - y_train (np.array) : Label on training data.
        - args_model.gdb_params (dict) : Hyperparamerter of GradientBoosting model
    Outputs:
        - model (GradientBoostingClassifier) : Trained model
    """

    cv = KFold(n_splits=args_model['cv_splits'], shuffle = True, random_state=args_model['random_state'])
    model = GradientBoostingClassifier(**args_model['gdb_params'])

    logger.info("[Model Training Steps] : Model Training...")
    model.fit(X_train, y_train)

    logger.info("[Model Training Steps] : Model Cross-validating...") 
    scores = cross_val_score(model, X_train, y_train, 
                            scoring = "accuracy", cv = cv, n_jobs=-1)
    
    logger.info("[Model Training Steps] : Model Validation")                     
    y_train_preds = model.predict(X_train)
    prc_train, rcl_train, fb_train = compute_model_metrics(y_train, y_train_preds)

    logger.info("[Model Training Steps] : Accuracy = %.3f (mean) %.3f (std)" % (mean(scores), std(scores)))
    logger.info("[Model Training Steps] : Precision: %.3f Recall: %.3f FBeta: %.3f" % (prc_train, rcl_train, fb_train))
    
    return model

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
    trained_model = train_model(feature_data, label_data, args_model)
    dump(trained_model, args_model['model_path'])


    

