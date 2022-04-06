import pytest
from fastapi.testclient import TestClient
from prediction_api import app

@pytest.fixture
def prediction_client():
    api_client = TestClient(app)
    return api_client

def test_get(prediction_client):
    response = prediction_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message" : "This is greeting page of applications !"}

def test_get_error(prediction_client):
    response = prediction_client.get("/none_url")
    assert response.status_code != 200

def test_post_class0(prediction_client):
    '''Test prediction on class 0 : salary <=50K'''
    response = prediction_client.post("/", json={
                            "age": 55,
                            "workclass": "Private",
                            "fnlgt": 190257,
                            "education": "HS-grad",
                            "marital_status": "Married-civ-spouse",
                            "occupation": "Farming-fishing",
                            "relationship": "Husband",
                            "race": "White",
                            "sex": "Male",
                            "hours_per_week": 53,
                            "native_country": "United-States"
                            }
    )
    assert response.status_code == 200
    assert response.json() == {"prediction" : "<=50K"}

def test_post_class1(prediction_client):
    '''Test prediction on class 1 : salary <=50K'''
    response = prediction_client.post("/", json={
                              "age": 27,
                              "workclass": "Federal-gov",
                              "fnlgt": 196386,
                              "education": "Assoc-acdm",
                              "marital_status": "Married-civ-spouse",
                              "occupation": "Adm-clerical",
                              "relationship": "Husband",
                              "race": "White",
                              "sex": "Male",
                              "hours_per_week": 40,
                              "native_country": "Other"
                            }
    )
    assert response.status_code == 200
    assert response.json() == {"prediction" : "<=50K"}

def test_post_error_value(prediction_client):
    response = prediction_client.post("/", json={
                              "age": 27,
                              "workclass": "Federal-gov",
                              "fnlgt": 196386,
                              "education": "None-Exist Value",
                              "marital_status": "Married-civ-spouse",
                              "occupation": "Adm-clerical",
                              "relationship": "Husband",
                              "race": "White",
                              "sex": "Male",
                              "hours_per_week": 40,
                              "native_country": "Other"
                            }
    )
    assert response.status_code == 422

def test_post_error_schema(prediction_client):
    response = prediction_client.post("/", json={
                              "age": 27,
                              "workclass": "Federal-gov",
                              "fnlgt": 196386,
                              "education": "Assoc-acdm",
                              "marital_status": "Married-civ-spouse",
                              "occupation": "Adm-clerical",
                              "relationship": "Husband",
                              "race": "White",
                              "sex": "Male",
                              "MistakeColumns": 40,
                              "native_country": "Other",
                            }
    )
    assert response.status_code == 422