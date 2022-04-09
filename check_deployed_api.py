import requests
import pandas as pd
import pprint

API_URL = "https://census-income-prediction-gb.herokuapp.com/"

test_data = pd.read_csv("./data/clean_test_data.csv")
json_test_data = test_data.to_dict(orient = 'records')

print("TEST : POST")
for data in json_test_data[0:1]:
  pprint.pprint(data)
  print("Actual value : ",data.pop("salary"))
  response_post = requests.post(API_URL, json = data)
  print("Predicted value : ", response_post.json())
  print("")

print("TEST : GET")
response_get = requests.get(API_URL)
print("GET resonse :", response_get.json())