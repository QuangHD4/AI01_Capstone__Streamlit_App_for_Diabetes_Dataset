import streamlit as st

data_expl_page = st.Page('pages/dataset_exploration.py', title= 'Dataset Exploration', icon=':material/data_exploration:')
intro_page = st.Page('pages/hello.py', title='Start Here', icon=':material/rocket_launch:')

st.navigation(pages=[intro_page, data_expl_page]).run()