import csv, json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

data_diabetes = pd.read_csv('data/diabetes.csv')

features = data_diabetes.drop(columns='Outcome')
label = data_diabetes['Outcome']

# stratified_sampling might work, cuz random_state changes the confusion matrix, 
#   and there's an imbalance in the number of examples for each class
# it really does, see the results below for urself 

for col in features.columns:
    Q1 = features[col].quantile(0.25)
    Q3 = features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    features[col] = features[col].clip(lower=lower, upper=upper)


# Split first
X_train, X_test, y_train, y_test = train_test_split(
    features, label, test_size = .1, random_state=0
)

logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('LogReg', LogisticRegression())
])
logreg_pipe.fit(X_train, y_train)
joblib.dump(logreg_pipe, 'models/LogReg.pkl')


knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('KNN', KNeighborsClassifier())
])
param_grid = {
    'KNN__n_neighbors': [i for i in range(1,13)],
    'KNN__weights': ['uniform', 'distance'],
    'KNN__metric': ['euclidean', 'manhattan', 'minkowski']
}
grid_search = GridSearchCV(
    knn_pipe,
    param_grid,
    cv=7,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

knn = grid_search.best_estimator_
joblib.dump(knn, 'models/knn.pkl')

model_preds = [logreg_pipe.predict(X_test), knn.predict(X_test)]
performance = [
    {
        'Accuracy':accuracy_score(y_test, model_pred), 
        'Precision':recall_score(y_test, model_pred), 
        'Recall':precision_score(y_test, model_pred), 
        'F1 score':f1_score(y_test, model_pred)
    } for model_pred in model_preds
]
with open('models/recorded_performance.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=tuple(performance[0].keys()))
    writer.writeheader()
    writer.writerows(performance)

# Careful: keys in models must match the name of the steps in pipelines
models = {'LogReg':logreg_pipe, 'KNN':knn_pipe}
configs = {name: models[name].named_steps[name].get_params() for name in models}
with open('models/final_configurations.json', 'w') as file:
    json.dump(configs, file, indent=4)

metric = 'accuracy'
base_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('LogReg', LogisticRegression(**configs['LogReg']))
])
base_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('KNN', KNeighborsClassifier(**configs['KNN']))
])

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=50, random_state=42)

scores_logreg = tuple(cross_val_score(base_logreg, X_train, y_train, cv=cv))
scores_knn = tuple(cross_val_score(base_knn, X_train, y_train, cv=cv))
kfoldcv_scores = [
    {
    'LogReg': scores_logreg[i], 
    'KNN': scores_knn[i]
    } for i in range(len(scores_logreg))
]

with open('models/kfoldcv_scores.csv', 'w') as file:
    writer = csv.DictWriter(file, fieldnames=['LogReg', 'KNN'])
    writer.writeheader()
    writer.writerows(kfoldcv_scores)