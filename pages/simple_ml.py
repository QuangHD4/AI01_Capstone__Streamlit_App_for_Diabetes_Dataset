import json

from src.utils import plot_cv_dist_with_ci

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ttest_rel

model = joblib.load('models/logreg.pkl')

model_tryout, code_and_model_selection = st.tabs(['Model Prediction', 'Code'])
with model_tryout:
    '''
    > You can try the final model (logistic regression) here. 
    If you need the value ranges for reference, refer back to the Data Exploration page
    '''
    with st.form('input features'):
        features = {
            'Pregnancies':None, 'Glucose':None, 'BloodPressure':None, 'SkinThickness':None, 
            'Insulin':None, 'BMI':None, 'DiabetesPedigreeFunction':None, 'Age':None
        }
        cols = list(features.keys())
        ll, lc, rc, rr = st.columns(4)
        with ll:
            features['Pregnancies'] = st.number_input('Pregnancies', min_value=0, step=1, format='%d', value=None, placeholder='')
            features['Insulin'] = st.number_input('Insulin', min_value=0, step=1, format='%d', value=None, placeholder='')
        with lc:
            features['Glucose'] = st.number_input('Glucose', min_value=0, step=1, format='%d', value=None, placeholder='')
            features['BMI'] = st.number_input('BMI', min_value=0.0, format='%.2f', value=None, placeholder='')
        with rc:
            features['BloodPressure'] = st.number_input('BloodPressure', min_value=0, step=1, format='%d', value=None, placeholder='')
            features['DiabetesPedigreeFunction'] = st.number_input('DiabetesPedigreeFunction', min_value=0.0, format='%.3f', value=None, placeholder='')
        with rr:
            features['SkinThickness'] = st.number_input('SkinThickness', min_value=0, step=1, format='%d', value=None, placeholder='')
            features['Age'] = st.number_input('Age', min_value=0, step=1, format='%d', value=None, placeholder='')
        submitted = st.form_submit_button('Get prediction')
    if submitted:
        null_features = [feature for feature in features.keys() if features[feature] is None]
        if null_features:
            st.error(f'Please fill in all input fields. {null_features}')
        else:
            model_input = [list(features.values())]
            pred = model.predict(model_input)[0]
            if pred:
                'Positive (have diabetes)'
            else:
                'Negative'

with code_and_model_selection:      # The other tab
    with st.expander('Full code'):
        code = (
            'features = data_diabetes.drop(columns=\'Outcome\')\n' 
            'labels = data_diabetes[\'Outcome\']\n\n'
            'for col in features.columns:\n'
            '    Q1 = features[col].quantile(0.25)\n'
            '    Q3 = features[col].quantile(0.75)\n'
            '    IQR = Q3 - Q1\n'
            '    lower = Q1 - 1.5 * IQR\n'
            '    upper = Q3 + 1.5 * IQR\n'
            '    features[col] = features[col].clip(lower=lower, upper=upper)\n\n'
            'features = StandardScaler().fit_transform(features)\n\n'
            'X_train, X_test, y_train, y_test = train_test_split(\n'
            '    features, labels, test_size = .1, random_state = 0\n'
            ')\n\n'
            'logreg_pipe = Pipeline([\n'
            '    (\'scaler\', StandardScaler()),\n'
            '    (\'LogReg\', LogisticRegression())\n'
            '])\n'
            'logreg_pipe.fit(X_train, y_train)\n\n'

            'knn_pipe = Pipeline([\n'
            '    (\'scaler\', StandardScaler()),\n'
            '    (\'KNN\', KNeighborsClassifier())\n'
            '])\n'
            'param_grid = {\n'
            '    \'KNN__n_neighbors\': [i for i in range(1,13)],\n'
            '    \'KNN__weights\': [\'uniform\', \'distance\'],\n'
            '    \'KNN__metric\': [\'euclidean\', \'manhattan\', \'minkowski\']\n'
            '}\n'
            'grid_search = GridSearchCV(\n'
            '    knn_pipe,\n'
            '    param_grid,\n'
            '    cv=7,\n'
            '    scoring=\'accuracy\',\n'
            ')\n'
            'grid_search.fit(X_train, y_train)\n'
        )
        st.code(code)

    '### 1. Data preprocessing'
    '''As mentioned in the previous page, there are many "impossible zeros", as well as outliers that needs to be treated. 
    We'll use the clipping method (fix the value range to the interquartile range) to address both of these problems at the same time.'''
    'Also, since the features have different value ranges, we\'ll scale them to improve convergence.'
    code = (
        'features = data_diabetes.drop(columns=\'Outcome\')\n' 
        'labels = data_diabetes[\'Outcome\']\n\n'
        'for col in features.columns:\n'
        '    Q1 = features[col].quantile(0.25)\n'
        '    Q3 = features[col].quantile(0.75)\n'
        '    IQR = Q3 - Q1\n'
        '    lower = Q1 - 1.5 * IQR\n'
        '    upper = Q3 + 1.5 * IQR\n'
        '    features[col] = features[col].clip(lower=lower, upper=upper)\n\n'
        'features = StandardScaler().fit_transform(features)\n\n'
        'X_train, X_test, y_train, y_test = train_test_split(\n'
        '    features, labels, test_size = .1, random_state = 0\n'
        ')'
    )
    if st.checkbox('Show code', key='preprocessing'):
        st.code(code)

    '### 2. Training'
    'For convenience, from this point onwards, I\'ll refer to Logistic Regression model as LogReg and K Nearest Neighbors model as KNN.'
    code = (
        'logreg_pipe = Pipeline([\n'
        '    (\'scaler\', StandardScaler()),\n'
        '    (\'LogReg\', LogisticRegression())\n'
        '])\n'
        'logreg_pipe.fit(X_train, y_train)\n\n'

        'knn_pipe = Pipeline([\n'
        '    (\'scaler\', StandardScaler()),\n'
        '    (\'KNN\', KNeighborsClassifier())\n'
        '])\n'
        'param_grid = {\n'
        '    \'KNN__n_neighbors\': [i for i in range(1,13)],\n'
        '    \'KNN__weights\': [\'uniform\', \'distance\'],\n'
        '    \'KNN__metric\': [\'euclidean\', \'manhattan\', \'minkowski\']\n'
        '}\n'
        'grid_search = GridSearchCV(\n'
        '    knn_pipe,\n'
        '    param_grid,\n'
        '    cv=7,\n'
        '    scoring=\'accuracy\',\n'
        ')\n'
        'grid_search.fit(X_train, y_train)\n'
    )
    if st.checkbox('Show code', key='training'):
        st.code(code)

    '##### 2.1. Final models\' configurations'
    with open('models/final_configurations.json', 'r') as file:
        models_configs = json.load(file)
    left_col, right_col = st.columns(2)
    with left_col:
        'LogReg'
        logreg_df = pd.DataFrame.from_dict(models_configs['LogReg'], orient='index', columns=['value'])
        logreg_df.index.name = 'parameter'
        st.dataframe(logreg_df, width=200)
    with right_col:
        'KNN'
        knn_df = pd.DataFrame.from_dict(models_configs['KNN'], orient='index', columns=['value'])
        knn_df.index.name = 'parameter'
        st.dataframe(knn_df, width=200)

    '##### 2.2. Models\' performance on test set'
    performance = pd.read_csv('models/recorded_performance.csv')
    performance.index = ['LogReg', 'KNN']
    st.write(performance)

    '##### 2.3. Model selection (why choose LogReg)'
    # 95% CI for the mean, see if they overlap substantially (using box plot?)
    # Perform pairec 2-sample hypothesis testing
    # Practical significance
    ''' To compare the performance of the 2 models used, we'll compare their accuracy scores. 
    Specifically, we'll compare the mean of each model's accuracy distribution (obtained
    by running repeated k-fold cross-validation for the base models)

    First, let's compute the 95% confidence interval for the mean accuracy of both models:
    '''
    kfoldcv_scores = pd.read_csv('models/kfoldcv_scores.csv')
    f'&emsp; `LogReg`: {np.mean(kfoldcv_scores['LogReg']):.4f} ± {1.96*np.std(kfoldcv_scores['LogReg']/(len(kfoldcv_scores['LogReg']))**0.5):.4f}'
    f'&emsp; `KNN`: {np.mean(kfoldcv_scores['KNN']):.4f} ± {1.96*np.std(kfoldcv_scores['KNN']/(len(kfoldcv_scores['KNN']))**0.5):.4f}'

    '''
    We can see that LogReg's confidence interval of the mean accuracy is higher and doesn't overlap with that of KNN, 
    which is a strong indication that it works slightly better for this task 
    '''

    with st.expander('*Extra: Visualizing performance distributions*'):
        fig = plot_cv_dist_with_ci(kfoldcv_scores)
        st.plotly_chart(fig)

    with st.expander('*Extra: Hypothesis testing*'):
        t_stat, p_val = ttest_rel(kfoldcv_scores['LogReg'], kfoldcv_scores['KNN'], alternative='greater')
        f'''
        Just as a little practice, let's go through the process of testing our previous hypothesis  
        Let:  
        - $H_1$ be *"The mean accuracy of the LogReg model is greater than that of the KNN model"* 
        (both models' hyperparams are as in section 2.1),  
        - $H_0$ will be *"The mean accuracy of the LogReg model is no greater than that of the KNN model"*

        Since we want to see if LogReg consistently outperforms KNN in folds (data is collected in pair for each fold during cross-validation), 
        we need to perform a paired 2-sample t-test  

        The test statistic is {t_stat:.3f} , and the corresponding p-value, assuming the significance level is 0.05, is {p_val:.2e}
        :material/arrow_right_alt: **Reject $H_0$, evidence favors $H_1$** (since p = {p_val:.2e} < 0.05)
        
        The result supports what we predicted earlier: LogReg beats KNN on accuracy for this dataset'''
        '---'
        st.caption('''
            A little note: Although the difference is statistically significant, it's not quite practically significant. 
            However, if this were to be used in medical diagnosis, a 1.3% difference, as well as LogReg's higher interpretability, 
            could make it be the better choice over KNN due to the high-stake nature of medical diagnosis
        ''')
