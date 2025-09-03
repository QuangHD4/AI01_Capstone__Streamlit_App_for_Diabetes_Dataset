import io, re, copy

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from src.elements import plot_charts, column_multicheck_dropdown_with_aggregations, df_info_table, build_sequence, clear_selections

# # inject custom css 
# st.markdown('''
#     <style>
#     a.custom-link {
#         color: white;
#         text-decoration: none;
#         font-size: 14px;
#         padding: 4px 8px;
#         border-radius: 8px;
#         transition: background-color 0.2s ease;
#         display: block;
#         margin-bottom: 4px;
#     }
#     a.custom-link:hover {
#         background-color: rgba(255, 255, 255, 0.2);
#     }
#     </style>
# ''', unsafe_allow_html=True)
# with st.sidebar:
#     st.markdown('''
#         <a class='custom-link' href='#1-dataset-overview'>1. Dataset overview</a>
#         <a class='custom-link' href='#2-univariate-analysis' >2. Univariate analysis</a>
#         <a class='custom-link' href='#3-bivariate-analysis'>3. Bivariate analysis</a>
#         <a class='custom-link' href='#3-1-correlation-matrix'>&emsp; 3.1. Correlation matrix</a>
#         <a class='custom-link' href='#3-2-feature-distribution-split-by-outcome'>&emsp; 3.2. Feature distribution split by outcome</a>
#         <a class='custom-link' href='#3-3-pairwise-scatter-plots'>&emsp; 3.3. Pairwise scatter plots</a>
#         <a class='custom-link' href='#4-playground'>4. Playground</a>
#         <a class='custom-link' href='#4-1-multivariate-analysis'>&emsp; 4.1. Multivariate analysis</a>
#         <a class='custom-link' href='#4-2-feature-engineering'>&emsp; 4.2. Feature engineering</a>
#         <a class='custom-link' href='#4-2-1-feature-preprocessing'>&emsp;&emsp; 4.2.1. Feature preprocessing</a>
#         <a class='custom-link' href='#4-2-2-feature-generation'>&emsp;&emsp; 4.2.2. Feature generation</a>
#         <a class='custom-link' href='#5-recap'>5. Recap</a>
#         <a class='custom-link' href='#6-references'>6. References</a>
#     ''', unsafe_allow_html=True)


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