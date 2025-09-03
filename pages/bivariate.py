import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

data_diabetes = pd.read_csv('data/diabetes.csv')
n_duplicates = data_diabetes.duplicated().sum()
data_diabetes.drop_duplicates(inplace=True)     # either use in-place or reassign


st.header('3. Bivariate analysis')
'''
In this stage, we primarily want to know how each feature relates to each other. 
Let's first grab a quick overview of the linear relationships among features 
with the correlation matrix
'''

st.subheader('3.1. Correlation matrix')
corr_matrix = data_diabetes.corr(method='pearson').round(3)
fig = px.imshow(corr_matrix, zmin = -1, zmax = 1, text_auto='.2f')          # add hover info for p values
st.plotly_chart(fig)
st.info('''
    - There's a moderate relationship between `Glucose` and `Outcome`, which suggests
    that `Glucose` might be an important feature in predicting diabetes in this 
    demographic (pima indian females from 21). Let's explore this relationship further in the next section.
    - Some feature pairs have weak to moderate correlation coefficient (e.g. `Age`-`Pregnancies`, 
    `Insulin`-`SkinThickness`, `BMI`-`SkinThickness`), which we'll need to keep in mind as 
    this could be a sign of multicollinearity or redundancy
''')
st.subheader('3.2. Feature distribution split by outcome')
selected = st.selectbox(
    'feature for split chart', 
    data_diabetes.columns.drop('Outcome'), 
    index=1,
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

'''
You can probably see that the group with diabetes tends to have higher glucose levels. 
Check out other features, as well.
'''

st.subheader('3.3. Pairwise scatter plots')
'''
We haven't seen any visualization of pairs of features yet. Let's make some 
scatter plots for this. Scatter plot is also a nice tool to see if there are non-linear 
relationships among feature pairs
'''
left_col, right_col = st.columns(2)
with left_col:
    feat_x = st.selectbox('x feature', data_diabetes.columns.drop('Outcome'), key='x_feat_2d')
with right_col:
    feat_y = st.selectbox('y feature', data_diabetes.columns.drop([feat_x, 'Outcome']), key='y_feat_2d')

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
