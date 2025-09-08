from flask import Flask
import pickle
import pandas as pd

app = Flask()


df = pd.DataFrame(
    [['M', '51-55', 7, 'B', "3", 1, 1]],
    columns=[
        'Gender',
        'Age',
        'Occupation',
        'City_Category',
        'Stay_In_Current_City_Years',
        'Marital_Status',
        'Product_Category_1'
    ]
)

# print(df)

with open(r"C:\Users\mhema\OneDrive\Desktop\DataScience\MachineLearning\regression.pkl", "rb") as file:
    pipeline = pickle.load(file)

print(pipeline.predict(df)[0])