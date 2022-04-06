import yaml
import pytest
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from src.data_cleaning import clean_data
from src.preprocess_data import process_data
from src.train_model import train_model

config = yaml.safe_load(open("./params.yml"))
data_config = config['data']
process_config = config['process_data']
train_config = process_config['train_mode']
model_config = config['model']

@pytest.fixture
def get_data():
    raw_data = pd.read_csv(config['data']['raw_data_path'])
    cleaned_data = clean_data(raw_data)
    feat_data, label_data, _, _ = process_data(
                                               cleaned_data,
                                               process_config['cat_features'],
                                               process_config['label'],
                                               train_config['train'],
                                               train_config['encoder'],
                                               train_config['label_bin'])
    return feat_data, label_data

def test_model_type(get_data):
    '''Check if model is GradientBoostingClassifier'''
    trained_model = train_model(get_data[0], 
                                get_data[1],
                                model_config)
    assert isinstance(trained_model, GradientBoostingClassifier)

def test_model_prediction(get_data):
    '''Check if model able to predict with selected features set'''
    trained_model = train_model(get_data[0], 
                                get_data[1],
                                model_config)

    try:
         trained_model.predict(get_data[0])
    except Exception as exc:
        assert False, "Trained model can't predict on existing features"

def test_binary_output(get_data):
    '''Check if model return only 2 class according to the binary problem'''
    trained_model = train_model(get_data[0], 
                                get_data[1],
                                model_config)
    predicted_value = trained_model.predict(get_data[0])
    assert len(set(predicted_value)) == 2

                    