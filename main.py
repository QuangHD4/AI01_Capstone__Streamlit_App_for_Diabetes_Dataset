import streamlit as st

intro_page = st.Page('pages/hello.py', title='Intro', icon=':material/rocket_launch:')
data_expl_page = st.Page('pages/dataset_exploration.py', title= 'Dataset Exploration', icon=':material/data_exploration:')
ml_page = st.Page('pages/simple_ml.py', title='Modeling', icon=':material/grain:')

st.navigation(pages={'':[intro_page], 'Main stuff':[data_expl_page, ml_page]}).run()
