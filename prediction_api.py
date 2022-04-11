import os
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from joblib import load

from src.preprocess_data import process_data
from src.model_validation import make_prediction
 
from pandas.core.frame import DataFrame
import numpy as np

class User(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    fnlgt: int
    education: Literal[
        'Doctorate', 'Assoc-voc', 'Prof-school','HS-grad',
        'Some-college','Bachelors','Assoc-acdm', 'School','Masters']
    marital_status: Literal[
        'Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse',
        'Widowed', 'Separated', 'Never-married', 'Divorced']
    occupation: Literal[
        'Prof-specialty', 'Priv-house-serv', 'Protective-serv',
        'Craft-repair', 'Farming-fishing', 'Adm-clerical',
        'Handlers-cleaners','Other-service', 'Sales',
        'Tech-support', 'Armed-Forces', 'Machine-op-inspct',
        'Transport-moving', 'Exec-managerial']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal['White', 'Black', 'Other']
    sex: Literal['Male', 'Female']
    hours_per_week: int
    native_country: Literal['United-States', 'Other']
    
    class Config:
        schema_extra = {
            "example" :{
                "age" : 27,
                "workclass": 'Federal-gov',
                "fnlgt": 196386,
                "education": 'Assoc-acdm',
                "marital_status": 'Married-civ-spouse',
                "occupation": 'Adm-clerical',
                "relationship": 'Husband',
                "race": 'White',
                "sex": 'Male',
                "hours_per_week": 40,
                "native_country": 'Other',
            }
        }

# For DVC making pull data on deployment
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("pip install 'dvc[s3]'")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global config, preprocess_config, preprocess_interf_config, trained_model, encoder_model, label_binarize
    
    config = yaml.safe_load(open("./params.yml"))
    preprocess_config = config['process_data']
    preprocess_interf_config = preprocess_config['interfence_mode']

    trained_model = load(config['model']['model_path'])
    encoder_model = load(preprocess_interf_config['encoder'])
    label_binarize = load(preprocess_interf_config['label_bin'])    

@app.get("/")
async def get_items():
    return {"message" : "This is greeting page of applications !"}

@app.post("/")
async def inference(user_data: User):
    input_array = np.array([[
        user_data.age,
        user_data.workclass,
        user_data.fnlgt,
        user_data.education,
        user_data.marital_status,
        user_data.occupation,
        user_data.relationship,
        user_data.race,
        user_data.sex,
        user_data.hours_per_week,
        user_data.native_country,
    ]])

    post_data_df = DataFrame(data = input_array, columns =[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours_per_week",
        "native_country",
    ])

    X_featurized, _, _, _ = process_data(post_data_df,
                                        categorical_features = preprocess_config['cat_features'],
                                        encoder=encoder_model,
                                        lb = label_binarize,
                                        training = preprocess_interf_config['train'])
                                        
    preds_value = make_prediction(trained_model, X_featurized)
    preds_value_decode = label_binarize.inverse_transform(preds_value)[0]
    return {"prediction" : preds_value_decode}
