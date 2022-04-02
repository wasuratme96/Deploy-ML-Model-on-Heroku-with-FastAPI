import yaml
import pytest
import pandas as pd
import numpy as np
from src.preprocess_data import process_data

config = yaml.safe_load(open("./params.yml"))
process_config = config['process_data']
train_config = process_config['train_mode']
inference_config = process_config['interfence_mode']

@pytest.fixture
def get_data():
    clean_data = clean_data = pd.read_csv(config['data']['clean_data_path'])
    return clean_data

@pytest.fixture
def get_featured_train_mode(get_data):
    feat_data, label_data, encoder, lb = process_data(
                                                      get_data,
                                                      process_config['cat_features'],
                                                      process_config['label'],
                                                      train_config['train'],
                                                      train_config['encoder'],
                                                      train_config['label_bin'])
    return feat_data, label_data, encoder, lb

def test_processed_data_shape(get_featured_train_mode):
    feat_data, label_data, _, _ = get_featured_train_mode
    assert len(feat_data) == len(label_data)

def test_label_data_typecheck(get_featured_train_mode):
    _, label_data, _, _ = get_featured_train_mode
    assert label_data.dtype == np.int

def test_label_data_valuecheck(get_featured_train_mode):
    _, label_data, _, _ = get_featured_train_mode
    assert len(set(label_data)) == 2

def test_encoder(get_data, get_featured_train_mode):
    _, _, encoder, _ = get_featured_train_mode
    categorical_data = get_data[process_config['cat_features']]
    try:
        encoder.transform(categorical_data)
    except Exception as exc:
        assert False, "Encoder can't transform existing data"


