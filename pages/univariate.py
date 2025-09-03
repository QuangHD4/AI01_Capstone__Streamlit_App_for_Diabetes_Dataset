import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import skew, kurtosis
import scipy.stats as stats
import matplotlib.pyplot as plt

from src.elements import column_multicheck_dropdown_with_aggregations

data_diabetes = pd.read_csv('data/diabetes.csv')
n_duplicates = data_diabetes.duplicated().sum()
data_diabetes.drop_duplicates(inplace=True)     # either use in-place or reassign

st.header('2. Univariate analysis')

left_col, right_col = st.columns(2)
with left_col:
    view = st.segmented_control(
        'select a tab', 
        ['Summary', 'Detail'], 
        default = 'Summary', 
        label_visibility='collapsed'
    )
with right_col:
    with st.container(horizontal_alignment='right'):
        selected = column_multicheck_dropdown_with_aggregations(data_diabetes)

if view == 'Summary':
    st.write(data_diabetes[selected].describe())
elif view == 'Detail':
    with st.container(border = True, height=600, gap=None):
        for pos, column in enumerate(selected):
            if pos != 0:
                st.divider()
            f'##### &ensp; **{column}**'
            l_hist, r_box = st.columns(2, gap = 'large')
            with l_hist:
                st.plotly_chart(px.histogram(data_diabetes, x=column, nbins=20, height=300))
            with r_box:
                if column == 'Outcome':
                    st.plotly_chart(px.pie(data_diabetes, names='Outcome', height=300))
                else:
                    st.plotly_chart(px.box(data_diabetes, y=column, height=300))
            if column == 'Glucose':
                st.info(f'''
                    - There are rows with value of 0, which is possibly missing values encoded as such since the glucose level can't be exactly 0
                    - The distribution (excluding the 0s) is moderately skewed to the right (skewness = {skew(data_diabetes.loc[data_diabetes[column]!=0, column]):.3f}). The shape looks like it could be a lognormal, but is cut off. Need more info on this.
                ''')
            elif column == 'BloodPressure':
                st.info(f'''
                    - There are rows with value of 0 (impossible)
                    - The distribution (excluding the 0s) is normal-like, with slighly heavy tails (kurtosis = {kurtosis(data_diabetes.loc[data_diabetes[column]!=0, column]):.3f})
                ''')
            elif column == 'SkinThickness':
                st.info(f'''
                    - There are a lot of 0s in this columns ({len(data_diabetes.loc[data_diabetes['SkinThickness']==0])/len(data_diabetes)*100:.2f}%). These might represent missing values
                    - There's an outlier with value = 99. This could be an error during measuring process.
                ''')
            elif column == 'Insulin':
                st.info(f'''
                    - There are a lot of impossible 0s ({len(data_diabetes.loc[data_diabetes['Insulin']==0])/len(data_diabetes)*100:.2f}%). They can be missing, but it can also be that their insulin level is really low (which is possible in some cases)
                    - The distribution (exclusing 0s and outliers) is visibly skewed to the right, likely indicating outliers
                ''')
            elif column == 'BMI':
                st.info(f'''
                    - There are 11 impossible 0s 
                    - The distribution, excluding the 0s, is slightly skewed to the right, probably due to outliers
                ''')
            elif column == 'DiabetesPedigreeFunction':
                st.info('The distribution is heavily skewed to the right and resembles a gamma distribution.')
                if st.checkbox('Show QQ plot'):
                    # Define gamma distribution parameters
                    alpha = 1.5
                    beta = 4
                    st.write(f'Comparing the data to a Gamma distribution with alpha = {alpha} and beta = {beta}')

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                    # --- Left Subplot: Gamma Distribution QQ Plot ---
                    shape, loc, scale = stats.gamma.fit(data_diabetes['DiabetesPedigreeFunction'])
                    stats.probplot(data_diabetes['DiabetesPedigreeFunction'], dist='gamma', sparams=(shape,), plot=ax1)
                    ax1.set_title('QQ Plot vs. Gamma Distribution')
                    ax1.set_xlabel('Theoretical Quantiles (Gamma)')
                    ax1.set_ylabel('Sample Quantiles (Data)')
                    ax1.grid(True)

                    # --- Right Subplot: Log-Normal Distribution QQ Plot ---
                    # Fit the log-normal distribution to the data to find optimal parameters
                    shape, loc, scale = stats.lognorm.fit(data_diabetes['DiabetesPedigreeFunction'])
                    stats.probplot(data_diabetes['DiabetesPedigreeFunction'], dist='lognorm', sparams=(shape,), plot=ax2)
                    ax2.set_title(f'QQ Plot vs. Log-Normal Distribution (shape={shape:.2f})')
                    ax2.set_xlabel('Theoretical Quantiles (Log-Normal)')
                    ax2.set_ylabel('Sample Quantiles (Data)')
                    ax2.grid(True)

                    # Adjust layout to prevent titles from overlapping
                    plt.tight_layout()

                    # Render the entire figure in Streamlit
                    st.pyplot(fig)
            elif column == 'Age':
                st.info('The distribution resembles an exponential distribution. The assumption is possible if we consider the total death rate to be constant')
            elif column == 'Outcome':
                st.info('The distribution of classes are good enough (an imbalanced dataset can have a minority class making up < 10-20% of the total dataset)')
nrows_sus_zero = (data_diabetes.iloc[:,1:6]==0).any(axis=1).sum()
st.info(f'''
    Summary of observations:  
    - There are 5 columns with impossible 0s. These are probably encoded missing values. 
        In total there are {nrows_sus_zero} rows ({nrows_sus_zero/len(data_diabetes)*100:.2f}%) 
        with one or more of these 0s. So we can't drop these; 
        instead, we'll have to replace them before building a predictive model
    - Outliers: There are probably some, but some extreme values in this dataset 
        are still less extreme than the highest documented ones. Further, 
        some features seems to follow a skewed distribution, resulting in 
        a large number of outliers in the box charts. We can naively 
        believe the example marked as outliers in the box charts to be true outliers, 
        but having domain knowledge would be more accurate.
    - Some of the features have normal-like distributions, and some doesn't. This information can be useful when selecting model that have assumptions about the distribution of features.
''')