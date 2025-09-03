import streamlit as st

overview = st.Page('pages/overview.py', title= 'Dataset Overview', icon=':material/data_exploration:')
univariate = st.Page('pages/univariate.py', title= 'Univariate Analysis', icon=':material/vital_signs:')
bivariate = st.Page('pages/bivariate.py', title= 'Bivariate Analysis', icon=':material/hive:')
playground = st.Page('pages/playground.py', title= 'Playground', icon=':material/view_in_ar:')

st.navigation(pages=[overview, univariate, bivariate, playground]).run()
