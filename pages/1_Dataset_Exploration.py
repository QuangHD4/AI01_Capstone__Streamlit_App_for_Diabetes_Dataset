import pandas as pd
import streamlit as st
import plotly.express as px

st.logo('assets/data_exploration_logo.png')
st.header('Data Exploration')

data_diabetes = pd.read_csv('data/diabetes.csv')

st.subheader('1. Dataset overview')

st.session_state

summary, detail, raw = st.tabs(['Summary', 'Detail', 'Raw data'])
with raw:
    data_diabetes
with summary:
    st.write(data_diabetes.describe())
with detail:
    if len(st.session_state) == 0:
        for column in data_diabetes.columns:
            st.session_state[column] = True
            
    selected = data_diabetes.columns.drop([key for key in st.session_state if not st.session_state[key]])

    with st.popover(f'*Showing {sum(st.session_state.values())} / {len(st.session_state)} columns*'):
        for column in data_diabetes.columns:
            st.checkbox(column, value = True, key=column)

    for pos, column in enumerate(selected):
        if pos != 0:
            st.divider()

        st.markdown(f'**{column}**')
        # TODO: write a short description and summary stats here
        
        hist, box = st.columns(2)
        with hist:
            st.plotly_chart(px.histogram(data_diabetes[column], nbins=30))
        with box:
            if column != 'Outcome':
                st.plotly_chart(px.box(data_diabetes[column]))

st.subheader('2. Visualization')
columns = data_diabetes.columns.drop('Outcome')

corr_matrix = data_diabetes.corr(method='pearson').round(3)
fig2 = px.imshow(corr_matrix, zmin = -1, zmax = 1)
st.plotly_chart(fig2)

left_col, right_col = st.columns(2)
with left_col:
    feat_x = st.selectbox('feature 1', columns)
with right_col:
    feat_y = st.selectbox('feature 2', columns.drop(feat_x), index=None, placeholder='choose a feature')

if feat_x and feat_y:
    st.write(f'Correlation coefficient: {corr_matrix.loc[feat_x, feat_y]}')
    st.scatter_chart(data_diabetes[[feat_x, feat_y]], x=feat_x, y=feat_y)

st.subheader('3. Observations')
('There seems to be an abnormal number of zero\'s for features '
 'that are not typically zero (e.g. glucose, blood pressure). '
 'This is likely the default value for missing data '
 'and needs to be addressed before performing more advanced analysis.'
)