import pandas as pd
import streamlit as st
import plotly.express as px

st.header('Data Exploration')

data_diabetes = pd.read_csv('data/diabetes.csv')

st.subheader('1. Dataset overview')

summary, detail, raw = st.tabs(['Summary', 'Detail', 'Raw data'])
with raw:
    data_diabetes
with summary:
    st.write(data_diabetes.describe())
with detail:
    if 'selected_cols' not in st.session_state:
        st.session_state['selected_cols'] = {column : True for column in data_diabetes.columns}
            
    selected = data_diabetes.columns.drop([key for key in st.session_state['selected_cols'] if not st.session_state['selected_cols'][key]])

    def toggle_col(name):
        st.session_state['selected_cols'][name] = bool(1 - st.session_state['selected_cols'][name])

    with st.popover(f'*Showing {sum(st.session_state['selected_cols'].values())} / {len(st.session_state['selected_cols'])} columns*'):
        for column in data_diabetes.columns:
            st.session_state['selected_cols'][column] = st.checkbox(column, value = True, on_change = toggle_col, args = [column])

    for pos, column in enumerate(selected):
        if pos != 0:
            st.divider()

        st.markdown(f'**{column}**')
        # TODO: write a short description and summary stats here
        
        hist, box = st.columns(2, gap = 'large')
        with hist:
            st.plotly_chart(px.histogram(data_diabetes[column], column, nbins=20))
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
    feat_y = st.selectbox('feature 2', columns.drop(feat_x))

if feat_x and feat_y:
    st.write(f'Correlation coefficient: {corr_matrix.loc[feat_x, feat_y]}')
    st.scatter_chart(data_diabetes[[feat_x, feat_y]], x=feat_x, y=feat_y)

st.subheader('3. Observations')
('There seems to be an abnormal number of zero\'s for features '
 'that are not typically zero (e.g. glucose, blood pressure). '
 'This is likely the default value for missing data '
 'and needs to be addressed before performing more advanced analysis.'
)