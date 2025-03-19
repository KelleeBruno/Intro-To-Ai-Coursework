# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 21:23:27 2025

@author: Kellee
"""

"""Extra stats for SVM"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

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


# Step 2: Bar Chart Distribution by Active/Passive Labels
active_counts = train['Binary'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=active_counts.index, y=active_counts.values, palette='viridis')
plt.title('Distribution of Active/Passive Labels')
plt.xlabel('Activity Status (0: Passive, 1: Active)')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Passive', 'Active'])
plt.show()

# Step 3: Pie Chart Distribution of Each Activity
activity_counts = train['ActivityName'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Activities')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.show()