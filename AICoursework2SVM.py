# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:52:03 2025

@author: Kellee
"""
# 1.1 Load the HAR data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

features = list()
with open('features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]

x_train = pd.read_csv('train/X_train.txt', delim_whitespace=True, header=None)
x_train.columns = features
x_train['subject'] = pd.read_csv('train/subject_train.txt', header=None).squeeze()
y_train = pd.read_csv('train/y_train.txt', names=['Activity']).squeeze()
y_trainlabels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',
                              4:'SITTING', 5:'STANDING',6:'LAYING'})
train = x_train.copy()
train['Activity'] = y_train
train['ActivityName'] = y_trainlabels

x_test = pd.read_csv('test/X_test.txt', delim_whitespace=True, header=None)
x_test.columns = features
x_test['subject'] = pd.read_csv('test/subject_test.txt', header=None).squeeze()
y_test = pd.read_csv('test/y_test.txt', names=['Activity']).squeeze()
y_testlabels = y_test.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',
                            4:'SITTING', 5:'STANDING',6:'LAYING'})
test = x_test.copy()
test['Activity'] = y_test
test['ActivityName'] = y_testlabels

train = shuffle(train)
test = shuffle(test)

# 1.2 Preprocess the data by reducing the number of features
pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=50)), ('svc', SVC())])

# 2.0 Convert the 6-class labels into binary (active vs inactive) problem
def to_binary_label(activity):
    if isinstance(activity, str):
        if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
            return 1
        else:
            return 0
    else:
        return None

train['Binary'] = train['ActivityName'].apply(to_binary_label)
test['Binary'] = test['ActivityName'].apply(to_binary_label)
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

# 3.0 Train baseline SVM models with different kernels (linear, polynomial, RBF)
x_train = train.drop(columns=['Activity', 'ActivityName', 'Binary', 'subject'])
y_train = train['Binary']
x_test = test.drop(columns=['Activity', 'ActivityName', 'Binary', 'subject'])
y_test = test['Binary']

def train_and_evaluate_svm(kernel):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('svc', SVC(C=100))])
    pipeline.fit(x_train, y_train)
    y_prediction = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    report = classification_report(y_test, y_prediction)
    print(f"Kernel: {kernel}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

for kernel in ['linear', 'poly', 'rbf']:
    train_and_evaluate_svm(kernel)

# 4.0 Perform hyperparameter tuning using GridSearchCV
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
param_grid = [
    {'svc__kernel': ['linear'], 'svc__C': [0.01, 0.1, 1, 10]},
    {'svc__kernel': ['poly'], 'svc__C': [0.01, 0.1, 1], 'svc__degree': [2, 3], 'svc__gamma': [0.001, 0.01, 0.1]},
    {'svc__kernel': ['rbf'], 'svc__C': [0.01, 0.1, 1, 10], 'svc__gamma': [0.001, 0.01, 0.1]}
]

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)

# 5.0 Evaluate and interpret results using confusion matrices and classification metrics
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_score = grid_search.best_score_
print("Best Cross-Validation Score:", best_score)
best_model = grid_search.best_estimator_

y_BestPrediction = best_model.predict(x_test)
cm = confusion_matrix(y_test, y_BestPrediction)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
print("Classification Report:")
print(classification_report(y_test, y_BestPrediction))
accuracy = accuracy_score(y_test, y_BestPrediction)
print(f"Accuracy: {accuracy:.4f}")