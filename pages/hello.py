import streamlit as st

print('new_run', '#'*100)

st.set_page_config(
    page_title='AI1 capstone project',
    page_icon='🚀'
)

st.header('👋 Hello there! Welcome to my project')
'''
This is an interactive streamlit app for exploring a diabetes dataset. We'll analyze the dataset 
and use this data to predict if a person has diabetes or not.  

#### 📦 The dataset
We'll be exploring the *Pima Indians Diabetes dataset* (available on 
[kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)), which includes some information of 
female patients in India.  

This dataset has 768 patients and 9 columns:  
- `Pregnancies`: Number of times pregnant  
- `Glucose`: Blood sugar level after a test  
- `BloodPressure`: Blood pressure when heart rests   
- `SkinThickness`: Arm skin thickness  
- `Insulin`: 2-hour serum insulin  
- `BMI`: Body Mass Index  
- `DiabetesPedigreeFunction`: Risk of diabetes based on family history  
- `Age`: In years  
- `Outcome`: 1 = diabetes, 0 = no diabetes  

#### 🎯 What we're doing with it (and how to explore)
- **Data Exploration page: Basic analysis** -- Summary statistics & Visualization of the dataset w/ `pandas` & `streamlit`  
- **Modelling page: Prediction with simple ML**  
    + *Data preprocessing*  
    + *Try 2 models*: Linear regression, clustering  
    :material/arrow_right_alt: Select better model, predict diabetes   
'''