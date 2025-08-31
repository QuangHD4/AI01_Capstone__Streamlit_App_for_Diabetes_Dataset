import os

from src.elements import multicol_hist_with_kde
from src.errors import *

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ttest_rel
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

st.set_page_config(layout='wide')

final_model = joblib.load('ml/models/SVC.pkl')

model_tryout, code_and_model_selection = st.tabs(['Model prediction', 'Training report'])
with model_tryout:
    columns = [             # for validation and display
        'Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age'
    ]
    input_method = st.radio(
        "Select your preferred input method:",
        ('Enter data manually', 'Upload a file'),
        horizontal=True
    )
    with st.form('input form', border=False):
        if input_method == 'Enter data manually':
            st.info(
                'Enter data for one or more patients. Click the :material/add: button to add rows.  \n\n' \
                'For best model performance, try to limit the value range to the training range. ' \
                'Anything outside of those ranges is extremely unlikely, if not humanly impossible, anyways.  \n' \
                'To see the training range of a column, hover over the header.'
            )

            input = st.data_editor(
                pd.DataFrame(columns=columns), 
                num_rows='dynamic', 
                column_config={
                    'Pregnancies': st.column_config.NumberColumn(
                        min_value=0,
                        step=1,
                        format='%d',
                        help='Training range: [0, 13]'
                    ),
                    'Glucose': st.column_config.NumberColumn(
                        min_value=0,
                        step=1,
                        format='%d',
                        help='Training range: [44, 199]\n\nHealthy range (approximate): [75, 140]'
                    ),
                    'BloodPressure': st.column_config.NumberColumn(
                        min_value=0,       # min value observed that falls under the distribution, ridiculously low in real life (below 60 and you can die)
                        step = 1,
                        help='Training range: [38, 106]\n\nTypical range: [60, 80]',
                        format='%d'
                    ),
                    'SkinThickness': st.column_config.NumberColumn(
                        min_value=0,
                        step=1,
                        help='Traning range: [0, 63]',
                        format='%d'
                    ),
                    'Insulin': st.column_config.NumberColumn(
                        min_value=0,
                        step=1,
                        help='Training range: [0, 318]\n\nTypical range: [15, 200]',
                        format='%d'
                    ),
                    'BMI': st.column_config.NumberColumn(
                        min_value=0.0,
                        help='Training range: [18.2, 50.0]\n\nTypical range: [16-severely underweight, 40-severely obese]',
                        format='%.1f'
                    ),
                    'DiabetesPedigreeFunction': st.column_config.NumberColumn(
                        min_value=0.0,
                        help='Traning range: [0.08, 2.42]',
                        format='%.3f'
                    ),
                    'Age': st.column_config.NumberColumn(
                        min_value=0,
                        help='Training range: [21, 81]',
                        format='%d'
                    )
                }
            )

        elif input_method == 'Upload a file':
            input = st.file_uploader("Choose a CSV file", type='csv')
            
        if st.form_submit_button('Get predictions'):
            st.divider()

            def get_prediction(input_df):
                input_df['Prediction'] = final_model.predict(input_df)
                def color_predictions(pred):
                    color = 'yellow' if pred else 'green'
                    return f'background-color: {color}'
                hilit_pred_df = input_df.style.map(color_predictions, subset=['Prediction'])       # LEARN: pandas styler
                st.dataframe(
                    hilit_pred_df,
                    column_config={
                        'BMI': st.column_config.NumberColumn(format='%.1f'),
                        'DiabetesPedigreeFunction': st.column_config.NumberColumn(format='%.3f'),
                    }
                )

            try:
                ref_df = pd.read_csv('data/diabetes.csv', nrows=5).drop(columns=['Outcome'])      # Isn't that's an intuitive way of comparing types? 
                valid_input, warnings = validate_input(input, ref_df=ref_df)
                for warning_msg in warnings:
                    st.warning(warning_msg)
            except NoFileSubmitted as e:
                st.error(e)
            except NoData as e:
                st.error(e)
            except NaNColumn as e:
                st.error(e)
            except TypeError as e:
                st.error(e)
            except MissingValue:
                st.error('Your input contains missing values. You can try fixing it yourself, or select one of the following quick fixes and click **Get Prediction** again ')
                fix_option = st.pills('What to do with missing values', options=['impute with mean', 'impute with median', 'drop'])        # refactor this?
                if fix_option:
                    get_prediction(quick_fix_na_val(input, fix_option))      # might raise error to fix the refactor
            else:
                get_prediction(valid_input)
                

with code_and_model_selection:      # The other tab

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

    '### &emsp; 2.1. Final models\' configurations'
    configs = {}
    saved_models = os.listdir('ml/models')
    for saved_model in saved_models:
        model_name = saved_model[:-4]       # remove .pkl
        params = joblib.load('ml/models/'+ saved_model).named_steps[model_name.lower()].get_params()
        configs[model_name] = '\n'.join([f'{k}: {v}' for k, v in params.items()])
    df_configs = pd.DataFrame.from_dict(configs, orient='index', columns=['Parameters'])
    st.dataframe(df_configs)

    '### &emsp; 2.2. Model selection'

    ''' To select the best model, we'll use accuracy as the main evaluating metric. 
    Specifically, we'll compare the mean of each model's accuracy distribution obtained
    by running repeated k-fold cross-validation for the base models (hyperparams are as in 2.1). 
    The results are scores of 2000 folds (10 splits, 200 repeats) with different random seeds  
    '''
    kfoldcv_scores = pd.read_csv('ml/scores/kfoldcv_scores.csv')

    '#### &emsp; &emsp; 2.2.1. Exploring performance data'
    with st.expander('*Visualizing performance distributions*'):
        multicol_hist_with_kde(kfoldcv_scores, xaxis_title='Accuracy')
        
    'Let\'s first examine the 95% CI of the mean accuracy of all models '
    means = [np.mean(kfoldcv_scores[model]) for model in kfoldcv_scores.columns]
    stds = [1.96 * np.std(kfoldcv_scores[model]) / (len(kfoldcv_scores[model])**0.5) for model in kfoldcv_scores.columns]
    ci95 = {model : [(
        f'{np.mean(kfoldcv_scores[model]):.4f} ± ' \
        f'{1.96 * np.std(kfoldcv_scores[model]) / (len(kfoldcv_scores[model])**0.5):.4f}'
    )] for model in kfoldcv_scores.columns}

    ci95 = pd.DataFrame.from_dict(ci95)
    st.dataframe(ci95, hide_index=True)

    ci95_fig = go.Figure()
    ci95_fig.add_trace(go.Scatter(
        x=kfoldcv_scores.columns,
        y=means,
        error_y=dict(
            type='data',
            array=stds,
            thickness=2,
            width=5
        ),
        mode='markers',
        marker=dict(size=10, color='royalblue'),
        name='Estimates'
    ))
    ci95_fig.update_layout(
        yaxis_title='Estimate',
        template='plotly_white'
    )
    st.plotly_chart(ci95_fig)

    '''
    Judging by the overlapping of CIs alone, it's likely that 
    `SVC` and `LogisticRegression` are the 2 candidates for the best model, and
    the performance rank from there is `RandomForestClassifier`, then 
    `GradientBoostingClassifier` and `MLPClassifier` at same place, and 
    at last place is `KNeighborsClassifier`.  

    To confirm this, let's test some hypotheses. 
    '''

    '#### &emsp; &emsp; 2.2.2. Hypothesis testing'

    st.markdown('''
    The plan here is perform a bunch of tests for every pair of models, just to be extra safe.  

    Let's go through an example to see how this goes. We'll consider `SVC` and 
    `LogisticRegression` for this test. Let:   
    - The ***alternative hypothesis*** be 
    *"The true mean accuracy of SVC is greater than that of LogisticRegression"* 
    (both models' hyperparams are as in section 2.1),  
                <p style='text-align: center;'>$H_1: \mu_{svc} > \mu_{lr}$</p>    

    - The ***null hypothesis*** be 
    *"The true mean accuracy of SVC is no greater than that of the LogisticRegression"*
                <p style='text-align: center;'>$H_0: \mu_{svc} = \mu_{lr}$</p> 
    ''', unsafe_allow_html=True)  
    st.caption(
        '''&emsp; &ensp;
        By convention, the null hypothesis uses the $=$ sign instead of $\le$. 
        The $\le$ is implicit in this case.
        '''
    )
    st.markdown(
        '''
        - The ***significance level***
                <p style='text-align: center;'>$\\alpha=0.01$</p> 
        ''', 
        unsafe_allow_html=True)
    '''
    Since we want to see if `SVC` consistently outperforms `LogisticRegression` in folds (data is collected in fold during cross-validation), 
    we need to perform a paired 2-sample t-test  
    '''

    if st.checkbox('Show maths'):
        pass     # TODO: add the formula for the paired t-test, use describe table to select mean and std for calc
    t_stat, p_val = ttest_rel(
        kfoldcv_scores['SVC'], kfoldcv_scores['LogisticRegression'],
        alternative='greater'
    )
    f'The test statistic $t_0= {t_stat:.4f}$ , and $p= {p_val:.4f}$, we therefore'
    _, center, _ = st.columns([1,2,1])
    with center:
        with st.container(border=True):
            f'**Reject $H_0$, evidence favors $H_1$** $(p = {p_val:.4f} < 0.01)$'
    
    '''
    The difference is statistically significant, meaning the 
    observed data is too rare to be obtained that there might likely be something 
    wrong with the null hypothesis, 
    and we can say that we are 99% sure that on average, 
    `SVC` performs *better** than `LogisticRegression` on this task
    '''
    st.caption(
        '''
        (*) The estimated difference, at about 0.0017, makes the difference not
        practically significant, but statistically significance basically says 
        "more is more ¯\\\\\_(ツ)_/¯"       
        '''                                                                     # idky, but it takes 5 backslashes to get the emoticon to work =))
    )
    st.divider()
    '''
    Let's now perform the rest of the tests. We'll keep the confidence level
    of 0.01 for all of them. Below is the summary of the results. 
    > A 🌝 means the row-model perform better statistically than the column-model, and
    a 🌚 means otherwise'''
    test_results = {column: [None for _ in kfoldcv_scores.columns] for column in kfoldcv_scores.columns}
    test_results = pd.DataFrame(test_results,index=kfoldcv_scores.columns)
    for y_model in kfoldcv_scores.columns:
        for x_model in kfoldcv_scores.columns:
            if x_model == y_model:
                continue

            _, p_val = ttest_rel(
                kfoldcv_scores[x_model], kfoldcv_scores[y_model], 
                alternative='greater'
            )
            test_results.loc[x_model, y_model] = '🌝' if p_val < 0.05 else '🌚'
    st.dataframe(test_results)
    st.markdown('''
    From this table, we can order the performance (with more confidence) as follows:  

    <p style='text-align:center;'>
        🥇 <code>SVC</code><br><br>
        🥈 <code>LogisticRegression</code> &emsp;&ensp; 🥉 <code>RandomForestClassifier</code><br><br>
        4️⃣ <code>MLPClassifier</code>, <code>GradientBoostingClassifier</code> &emsp;&ensp; 6️⃣ <code>KNeighborClassifier</code> 
    </p>

    This matches our predictions earlier. *yaaaay*
    ''', unsafe_allow_html=True)
    '''
    ---
    Except that, if you had checked the *Show maths* checkbox earlier, you might have noticed 
    that the test statistic $t_0$ and the p-value $p$ doesn't depend on our choice
    of the significance level $\\alpha$. That means we could have set it to be as high
    as $p$ after it has be calculated, right, right?  

    ❌ No, we *must* define it before we do any calculations, and as far as I know, 
    researchers often needs to pre-declare and justify the choice of $\\alpha$, 
    along with some other stuff, before even collecting data

    That's because a biased choice of $\\alpha$ can lead to potentially misleading conclusions.
    Imagine if our result for the example test with (`SVC`, `LogisticRegression`) 
    wasn't significant at $\\alpha=0.01$ but it would be if $\\alpha=0.05$. 
    I totally could have changed $\\alpha$ to 0.05 to make the conclusion seems plausible
    and hid the fact that I did. Even if I didn't change $\\alpha$, I could still have run
    a bunch of tests on different sample sizes and select one that would make my conclusion significant.

    This is an infamous problem in research, and there's already numerous ways
    to counteract it, including the method pre-declaration approach mentioned
    earlier, but there's one that I think is really interesting: ***Bayesian hypothesis testing***.
    '''
    with st.expander('*Extra: Bayesian hypothesis testing*'):
        '''
        Let's plot pairwise scatter plots to see if there's any consistent performance difference across folds'''

        selected_models = st.multiselect(
            'Select models to plot pairwise performance',
            options=kfoldcv_scores.columns,
            default=['LogisticRegression', 'SVC', 'RandomForestClassifier'],
            label_visibility='hidden'
        )
        pair_kfoldcv_score_fig = make_subplots(
            rows=len(selected_models), 
            cols=len(selected_models),
            row_titles=selected_models,
            column_titles=selected_models
        )
        kfoldcv_scores = kfoldcv_scores.round(4)
        for row_no, y_model in enumerate(selected_models):
            for col_no, x_model in enumerate(selected_models):
                if row_no == col_no:
                    continue
                counts = kfoldcv_scores.groupby([x_model, y_model]).size().reset_index(name='size')
                px.scatter(counts, x=x_model,y=y_model, size='size')
                pair_kfoldcv_score_fig.add_trace(
                    go.Scatter(
                        x=counts[x_model], y=counts[y_model],
                        mode='markers', 
                        marker=dict(
                            size=counts['size']*6/len(selected_models),
                            sizemode='area',
                        ),
                    ),
                    row=row_no+1, col=col_no+1,
                )
                pair_kfoldcv_score_fig.add_shape(
                    type="line",
                    x0=min(kfoldcv_scores[y_model].min(), kfoldcv_scores[x_model].min()),
                    y0=min(kfoldcv_scores[y_model].min(), kfoldcv_scores[x_model].min()),
                    x1=max(kfoldcv_scores[y_model].max(), kfoldcv_scores[x_model].max()),
                    y1=max(kfoldcv_scores[y_model].max(), kfoldcv_scores[x_model].max()),
                    line=dict(
                        color="yellow",
                        width=1.5,
                        # dash="dash"
                    ),
                    row=row_no+1, col=col_no+1,
                )
        pair_kfoldcv_score_fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(pair_kfoldcv_score_fig)

        '''
        You might observe that for plots where the x axis is one of the three better performing models that we suspect ealier (LogReg, Forest, SVC)
        and the y axis is one of the three worse performing ones, the half below the yellow line is denser, 
        which suggests that the better three already outperform the other three in the majority of the folds.  

        Nonetheless, the difference is not inherently obvious, and the proportion of folds 
        where the model with lower suspected performance outperforms those deemed "better" are not small in almost all pairs. 
        So, based on these basic analyses, we can't really tell the best performing model yet.
        '''
    st.caption('You can read, if you want. If hate maths, then you can pass ~')


    '### &emsp; 2.3. Models\' performance on test set'
    performance = pd.read_csv('ml/scores/test_performance.csv', index_col='Model')
    st.write(performance)

    '''
    ### 3. Limitations
    The training and evaluation process presented above is not at all what should be done. I
    just want to practice some stuff that I recently learned. Here's some of the most significant limitations:
    - Overly simplistic model selection framework
        - Insufficient metrics: Using accuracy alone is not a good way to compare models,
        especially for imbalanced dataset as this one
        - 
    - Parameter grid not extensive enough
    - dataset from 1 demographic -> not much generalizability
    - more advanced imputations
    - 
    

    
    ### Resource note  

    For the full code of generating the models and scores, check out
    `scripts/models_training.ipynb` in
    [my repo](https://github.com/QuangHD4/AI01_Capstone__Streamlit_App_for_Diabetes_Dataset/tree/master).
    You can also view the pre-trained models and their scores, 
    which are used in this page, in the `ml` folder.
    '''