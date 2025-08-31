from src.elements import plot_charts, column_multicheck_dropdown_with_aggregations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go



data_diabetes = pd.read_csv('data/diabetes.csv')

st.header('Data Exploration')
st.subheader('1. Dataset Overview')

left_col, right_col = st.columns([5,3])
with left_col:
    tab = st.segmented_control(
        'select a tab', 
        ['Summary', 'Detail', 'Raw data'], 
        default = 'Summary', 
        label_visibility='collapsed'
    )
with right_col:
    with st.container(horizontal_alignment='right'):
        selected = column_multicheck_dropdown_with_aggregations(data_diabetes)

match tab:
    case 'Raw data':
        data_diabetes[selected]
    case 'Summary':
        st.write(data_diabetes[selected].describe())
    case 'Detail':
        plot_charts(
            data=data_diabetes,
            cols=selected,
            type=['hist', 'box']
        )



st.subheader('2. Feature Relationships')

st.markdown('#### 2.1. Correlation matrix')
corr_matrix = data_diabetes.corr(method='pearson').round(3)
fig = px.imshow(corr_matrix, zmin = -1, zmax = 1, text_auto='.2f')
st.plotly_chart(fig)

st.markdown('#### 2.2. Feature distribution split by outcome')
selected = st.selectbox(
    'feature for split chart', 
    data_diabetes.columns.drop('Outcome'), 
    label_visibility='collapsed'
)

no_diabetes = data_diabetes[selected].loc[data_diabetes.Outcome == 0]
yes_diabetes = data_diabetes[selected].loc[data_diabetes.Outcome == 1]

nbins = (data_diabetes[selected].max() if selected == 'Pregnancies' else 25)
fig = ff.create_distplot(
    [no_diabetes, yes_diabetes], 
    ['Without diabetes', 'With diabetes'], 
    bin_size = (data_diabetes[selected].max() - data_diabetes[selected].min())/nbins,
    show_rug = False
)
st.plotly_chart(fig)

st.markdown('#### 2.3. Pairwise scatter plots')
left_col, right_col = st.columns(2)
with left_col:
    feat_x = st.selectbox('Feature 1 (x)', data_diabetes.columns.drop('Outcome'))
with right_col:
    feat_y = st.selectbox('Feature 2 (y)', data_diabetes.columns.drop([feat_x, 'Outcome']))

if st.checkbox('Split by outcome'):
    scatter_split_data = data_diabetes.copy()
    label_map = {0:'With diabetes', 1:'No diabetes'}
    scatter_split_data['Outcome'] = scatter_split_data['Outcome'].map(label_map)
    scatter_split_outcome_fig = px.scatter(
        scatter_split_data, x=feat_x,y=feat_y, color='Outcome',
        color_discrete_map={'With diabetes':'#D8EAF7', 'No diabetes':'#004CA4'},
        opacity=0.6, 
        trendline='ols'
    )
    scatter_split_outcome_fig.update_traces(marker={'size':9})
    st.plotly_chart(scatter_split_outcome_fig)
else:
    fig = px.scatter(x=data_diabetes[feat_x], y=data_diabetes[feat_y], trendline='ols')
    fig.update_traces(marker_size=10)
    fig.update_layout(
        xaxis_title=feat_x, yaxis_title=feat_y, 
        title=f'Correlation coefficient: {corr_matrix.loc[feat_x, feat_y]}')
    st.plotly_chart(fig)



st.subheader('3. Observations')
('- There seems to be an abnormal number of zero\'s for features '
 'that are not typically zero (e.g. `Glucose`, `BloodPressure`). '
 'This is likely the default value for missing data '
 'and needs to be addressed before performing more advanced analysis.'
)
('- The `Glucose` feature has relatively high correlation with respect to the Outcome, '
 'which might indicate this feature is important in predicting diabetes, '
 'though more analysis is needed to confirm this'
)
('- The number for each class is quite imbalance: 65% without diabetes, '
'nearly twice as much as the numbers of diabetes examples. '
''
)