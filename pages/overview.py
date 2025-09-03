import pandas as pd
import streamlit as st

from src.elements import df_info_table

data_diabetes = pd.read_csv('data/diabetes.csv')
n_duplicates = data_diabetes.duplicated().sum()
data_diabetes.drop_duplicates(inplace=True)     # either use in-place or reassign

st.header('1. Dataset Overview')
'''
We'll be exploring the *Pima Indians Diabetes dataset* (available on 
[kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)), which includes some information of 
female patients of Pima Indian heritage, living near Phoenix, Arizona, USA.   
'''
option = st.segmented_control(
    'raw/info', 
    options=['Raw data', 'Dataset info', 'Column description'], 
    default='Raw data',
    label_visibility='collapsed',
)
if option == 'Raw data':
    data_diabetes
elif option == 'Dataset info':
    f'Row count: {len(data_diabetes)} &emsp;&ensp; Unique row count: {len(data_diabetes)-n_duplicates}'
    df_info_table(data_diabetes)
elif option == 'Column description':
    '''
    - `Pregnancies`: Number of times pregnant  
    - `Glucose`: Blood sugar level after a test  
    - `BloodPressure`: Blood pressure when heart rests   
    - `SkinThickness`: Arm skin thickness  
    - `Insulin`: 2-hour serum insulin  
    - `BMI`: Body Mass Index  
    - `DiabetesPedigreeFunction`: Risk of diabetes based on family history  
    - `Age`: In years  
    - `Outcome`: 1 = diabetes, 0 = no diabetes  
    '''
st.info('''
    Observation: The dataset doesn't have missing values or duplicates  
''')