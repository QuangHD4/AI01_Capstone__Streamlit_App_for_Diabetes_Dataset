# AI01 Capstone Project: Diabetes Prediction Web App


# Overview
This project is a Streamlit-based web application for predicting the likelihood of diabetes in patients based on various health metrics. It uses machine learning models trained on the Pima Indians Diabetes Dataset. The app allows users to explore the dataset, understand relationships between features, and test predictions using different models.  

Deployed App: [Click here to try it out](https://ai01-capstone-quang-bk85f8nm8fonqyquqnenh4.streamlit.app/)

# Features in each page
- Dataset Exploration
  - View dataset overview, summary statistics, and single-feature visualizations.
  - Check correlations and scatter plots between features.
  - Read initial observations from the dataset.
- Modeling
  - Compare multiple models (Logistic Regression, KNN, Random Forest) using cross-validation.
  - View model configurations and performance metrics.
  - Predict diabetes likelihood by entering patient data.

# Dataset
Source: Pima Indians Diabetes Database, available on [kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)  
Attributes: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age  
Target: Outcome - diabetes (1), non-diabetes (0).

# Installation
1. Clone the repository
```
git clone https://github.com/QuangHD4/AI01_Capstone__Streamlit_App_for_Diabetes_Dataset.git
cd https://github.com/QuangHD4/AI01_Capstone__Streamlit_App_for_Diabetes_Dataset.git
```
2. Create and activate a virtual environment  
```python -m venv venv```  
Windows:  
```env\Scripts\activate```  
macOS/Linux:  
```source env/bin/activate```  
4. Install dependencies  
```pip install -r requirements.txt```  
5. Run the Streamlit app  
```streamlit run main.py```

# Project Structure
```
project-folder/
|-- main.py               # Main Streamlit app file
|-- requirements.txt      # Dependencies
|-- models/               # Trained models & related data
|-- data/                 # Dataset 
|-- src/                  # Helper scripts (data processing & modeling)
|-- README.md             # Project doc
```
