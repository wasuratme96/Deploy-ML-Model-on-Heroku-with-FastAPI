import os
from sklearn.preprocessing import label_binarize
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from joblib import load

from src.preprocess_data import process_data
from src.model_validation import inference
 
from pandas.core.frame import DataFrame
import numpy as np

class User(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    maritalStatus: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    hoursPerWeek: int
    nativeCountry: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

# For DVC making pull data on deployment
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

config = yaml.safe_load(open("./params.yml"))
preprocess_config = config['process_data']
preprocess_interf_config = preprocess_config['interfence_mode']
model_config = config['model']

@app.get("/")
async def get_items():
    return {"message" : "This is greeing page of applications !"}

@app.post("/")
async def inference(user_data: User):
    trained_model = load(config['model']['model_path'])
    encoder_model = load(preprocess_interf_config['encoder'])
    label_binarize = load(preprocess_interf_config['label_bin'])

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
        user_data.salary
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
        "salary"
    ])

    X_featurized, _, _, _ = process_data(post_data_df,
                                        categorical_features = preprocess_config['cat_features'],
                                        encoder=encoder_model,
                                        lb = label_binarize,
                                        training = preprocess_interf_config['train'])
    preds_value = inference(trained_model, X_featurized)
    preds_value_decode = label_binarize.inverse_transform(preds_value)[0]
    return {"prediction" : preds_value_decode}
