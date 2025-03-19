# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 21:03:23 2025

@author: Kellee
"""

"""Basic statistics and graphs for SVM and linear regression"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

Cali_Housing_Data = fetch_california_housing()
Dataset = pd.DataFrame(data=Cali_Housing_Data.data, columns=Cali_Housing_Data.feature_names)
Dataset['MedHouseVal'] = Cali_Housing_Data.target

#heatmap
plt.figure()
correlation_matrix = Dataset.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='BuPu', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap')
plt.show()

#pdf
plt.figure(figsize=(10, 6))
sns.histplot(Dataset['MedHouseVal'], bins=30, kde=True)
plt.title('Probability Distribution of Median House Value')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.show()

#Boxplot
plt.figure(figsize=(12, 5))
sns.boxplot(x=Dataset['MedInc'])
plt.title('Box Plot of Median Income (scaled to $10,000s)')
plt.xlabel('Median Income')
plt.show()