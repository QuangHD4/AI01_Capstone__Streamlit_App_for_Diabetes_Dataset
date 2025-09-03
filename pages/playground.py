import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from src.elements import build_sequence, clear_selections

data_diabetes = pd.read_csv('data/diabetes.csv')
n_duplicates = data_diabetes.duplicated().sum()
data_diabetes.drop_duplicates(inplace=True)     # either use in-place or reassign

st.header('4. Playground')
st.markdown('''
    The sections above covered some essential stuff when exploring a dataset 
    (except Data cleaning, which we'll cover in section 4.2.1). 
    In this section (4), you can explore extra steps that aren't necessarily required 
    (at least that's as fas as I know)
''', unsafe_allow_html=True)
st.markdown('### 4.1. Multivariate analysis')
left_col, center_col, right_col = st.columns(3)
with left_col:
    x = st.selectbox('x feature', data_diabetes.columns.drop('Outcome'))
with center_col:
    y = st.selectbox('y feature', data_diabetes.columns.drop(['Outcome', x]))
with right_col:
    z = st.selectbox('z feature', data_diabetes.columns.drop(['Outcome', x, y]))
if st.checkbox('Split by outcome', key='split_outcome_3d'):
    scatter_split_data = data_diabetes.copy()
    label_map = {0:'With diabetes', 1:'No diabetes'}
    scatter_split_data['Outcome'] = scatter_split_data['Outcome'].map(label_map)
    scatter_3d_fig = px.scatter_3d(
        scatter_split_data, x=x, y=y, z=z, color='Outcome',
        color_discrete_map={'With diabetes':'#E4CD4D', 'No diabetes':'#1B85FF'},
        opacity=0.5)
else:
    scatter_3d_fig = px.scatter_3d(data_diabetes, x=x, y=y, z=z, opacity=0.3)
scatter_3d_fig.update_traces(marker=dict(size=5))
st.plotly_chart(scatter_3d_fig)

st.markdown('### 4.2. Feature engineering')
st.markdown('#### &emsp;4.2.1. Feature preprocessing')
'''
##### &emsp;&emsp;&emsp;Handling sus 0s
As previously discussed, there are 5 columns that can't have 0s as values, 
but we can't afford to drop all of the examples containing these 0s, since they
make up nearly half of our dataset. A suitable solution is treating 0s as missing 
and using imputation, i.e. filling in these 0s with some other values. 
You can explore a few imputation approach provided below.'''

cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputation_method = st.selectbox(
    'Select an imputation method:',
    ('Mean', 'Median', 'KNNImputer', 'IterativeImputer')
)

# choose view
show_orig = st.toggle('Show original')
df_display = data_diabetes.copy()
df_imputed = data_diabetes.copy()
df_imputed[cols_to_impute] = df_imputed[cols_to_impute].replace(0, np.nan)
if not show_orig:
    if imputation_method == 'Mean':
        for col in cols_to_impute:
            mean_val = df_imputed[col].mean()
            df_imputed[col].replace(np.nan, mean_val, inplace=True)
    
    elif imputation_method == 'Median':
        for col in cols_to_impute:
            median_val = df_imputed[col].median()
            df_imputed[col].replace(np.nan, median_val, inplace=True)
            
    elif imputation_method == 'KNNImputer':
        imputer_knn = KNNImputer(n_neighbors=5)
        df_imputed[cols_to_impute] = imputer_knn.fit_transform(df_imputed[cols_to_impute])
        
    elif imputation_method == 'IterativeImputer':
        imputer_iter = IterativeImputer(max_iter=10, random_state=0)
        df_imputed[cols_to_impute] = imputer_iter.fit_transform(df_imputed[cols_to_impute])
    df_display = df_imputed

#plots
if not show_orig:
    plot_title = f'Distributions after {imputation_method} Imputation'
else:
    plot_title = 'Original Distributions (with 0s removed for better comparison of bins)'
    df_display[cols_to_impute] = df_display[cols_to_impute].replace(0, np.nan)

# show plot
fig = px.histogram(df_display, x=cols_to_impute, facet_col="variable", facet_col_wrap=3,
                   title=plot_title)
for axis in fig.layout:
    if axis.startswith('xaxis') or axis.startswith('xaxis2'):
        fig.layout[axis].matches = None
st.plotly_chart(fig)

'##### &emsp;&emsp;&emsp;Transforming feature distribution'
'''
Here, you can test some transformations on the features. This is a common 
practice for improving predictive model performance. I haven't made a model 
to see this in effect in training, though you can still preview the effects 
of the transformations on these features.

To get started, 
try using the log transformation, or a power transformation with a 
positive number that's less than 1 on the skewed `DiabetesPedigreeFunction`
to make it more normal-like. 
'''
column = st.selectbox('Choose feature to transform', options=data_diabetes.columns.drop('Outcome'), index=6)
all_transforms = {'exp':np.exp, 'log':np.log1p, 'pow':np.power, 'mul':np.multiply}
transform_params = {
    'pow': {
        'type': 'slider',
        'label': 'to the power of',
        'min': -3.0,
        'max': 3.0,
        'default': 1.0,
        'step': 0.01,
        'label_visibility': 'collapsed'
    },
    'mul': {
        'type': 'slider',
        'label': 'multiply by',
        'min':0.01,
        'max':10.0,
        'default':1.0,
        'step':0.01,
        'label_visibility': 'collapsed'
    }
}
if "preview" not in st.session_state:
    st.session_state.preview = False
left_col, right_col = st.columns([0.27,0.73], gap='medium')
with left_col:
    with st.container(height=450, border=None):
        st.caption('Transform sequence :material/arrow_cool_down:')
        transform_sequence, params = build_sequence(list(all_transforms.keys()), follow_up_config=transform_params, item_caption='Transform')
        if len(transform_sequence) >= 1:
            st.session_state.preview = True
        else:
            st.session_state.preview = False
        if st.button('Reset'):
            clear_selections()
    with right_col:
        view_orig = st.toggle('original')
        if st.session_state.preview and not view_orig:      # preview mode
            # apply transform
            data_diabetes[f'Transformed_{column}'] = data_diabetes[column]
            for id, type in enumerate(transform_sequence):
                if type in transform_params:
                    param = params[id]
                    data_diabetes[f'Transformed_{column}'] = all_transforms[type](data_diabetes[f'Transformed_{column}'], param)
                else:
                    data_diabetes[f'Transformed_{column}'] = all_transforms[type](data_diabetes[f'Transformed_{column}'])

            fig = px.histogram(data_diabetes, x=f'Transformed_{column}', color='Outcome')
        else: 
            fig = px.histogram(data_diabetes, x=column, color='Outcome')
        fig.update_layout(title='Preview')
        st.plotly_chart(fig)
# if 'applied_transforms' not in st.session_state:
#     st.session_state.applied_transforms = {}
# left_col, right_col = st.columns([0.3,0.7])
# with left_col:
#     if st.button('Apply this transform'):
#         st.session_state.applied_transforms[column] = (
#             copy.deepcopy(st.session_state.transforms), 
#             copy.deepcopy(st.session_state.follow_ups)
#         )
#         'Transformation applied to data'
# with right_col:
#     if st.button('Reset transforms'):
#         st.session_state.applied_transforms = []

# fig_log = ff.create_distplot([data_diabetes['DiabetesPedigreeFunction']], group_labels=[''], histnorm='probability density', bin_size=.05, show_rug=False)
# st.plotly_chart(fig_log)

# '''
# ##### Handling outliers
# If we assume the examples with values outside the range of (Q1-1.5*IQR, Q3+1.5*IQR),
# '''

st.markdown('#### &emsp;4.2.2. Feature generation')
'''
We can create new features from existing ones to help the predictive model (if
we make one). Keep in mind that the interpretability of these new features is 
important, as it can help us understand why the model predicts the way it does.
'''

'##### &emsp;&emsp;&emsp;Binning (Create categorical data from numerical data)'
left_col, center_col ,right_col = st.columns([1,1,2], gap='large')
with left_col:
    column = st.selectbox('Column to bin', options=data_diabetes.columns.drop('Outcome'), index=5)
with center_col:
    num_bins = st.slider('Number of bins:', min_value=2, max_value=20, value=5)
data_diabetes[f'{column}_Category'] = pd.cut(data_diabetes[column], bins=num_bins, right=False)
data_diabetes[f'{column}_Category'] = data_diabetes[f'{column}_Category'].astype(str)
data_diabetes[f'{column}_Category'] = data_diabetes[f'{column}_Category'].astype(str).str.replace(r'[\[()\]]', '', regex=True).str.replace(r', ', '-', regex=True)
sorted_categories = sorted(data_diabetes[f'{column}_Category'].unique(), key=lambda x: float(x.split('-')[0]))
with right_col:
    other_feat = st.selectbox(
        'Column to see interaction with binned feature',
        options=data_diabetes.columns.drop([column, f'{column}_Category', 'Outcome']),
        index=None
    )
st.divider()

if other_feat:
    fig = px.box(data_diabetes, x=f'{column}_Category', y=other_feat)
else:
    fig = px.histogram(data_diabetes, x=f'{column}_Category', color='Outcome',
                            title=f'Distribution of {column} Category',
                            category_orders={f'{column}_Category': sorted_categories})
st.plotly_chart(fig)

'''
#### &emsp;&emsp;&emsp;Interaction features  

There are a few ways to create interaction features. A simple approach is to make
a linear combination of features. Try it out below.
'''

col1, col2 = st.columns(2)
with col1:
    feat_1 = st.selectbox('First feature', data_diabetes.columns.drop('Outcome'), label_visibility='collapsed')
    coeff_1 = st.number_input(f'Coefficient for {feat_1}:', value=1.0)
with col2:
    feat_2 = st.selectbox('Second feature', data_diabetes.columns.drop([feat_1,'Outcome']), label_visibility='collapsed')
    coeff_2 = st.number_input(f'Coefficient for {feat_2}:', value=1.0)

data_diabetes[f'{feat_1}_{feat_2}_Combined'] = (coeff_1 * data_diabetes[feat_1]) + (coeff_2 * data_diabetes[feat_2])

test_feat = st.selectbox('Comparing feature', data_diabetes.columns.drop(['Outcome']))
# Plot the scatter plot
fig_scatter = px.scatter(data_diabetes, x=f'{feat_1}_{feat_2}_Combined', y=test_feat, color='Outcome',
                         labels={'Glucose_BMI_Combined': f'({coeff_1:.2f} * Glucose) + ({coeff_2:.2f} * BMI)'})
st.plotly_chart(fig_scatter)
