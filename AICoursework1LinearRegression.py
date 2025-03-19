# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:15:10 2025

@author: Kellee
"""
"""AI Coursework"""

#1.1 Load the dataset using sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Cali_Housing_Data = fetch_california_housing()
Dataset = pd.DataFrame(data=Cali_Housing_Data.data, columns=Cali_Housing_Data.feature_names)

#1.2 Plot a scatter plot of MedInc vs MedHouseVal
Dataset['MedHouseVal'] = Cali_Housing_Data.target
plt.figure()
plt.scatter(Dataset['MedInc'], Dataset['MedHouseVal'], s=0.01)
plt.title('Median Income vs Median House Value')
plt.xlabel('Median Income in 10K USD')
plt.ylabel('Median House Value in 100K USD')
plt.show()

#1.3 Calculate summary statistics for both variables
summary_statistics = Dataset.describe()
print(summary_statistics)

#2.1 Split data into training and testing 80/20
Targetvar = Dataset['MedHouseVal']
Features = Dataset.drop('MedHouseVal', axis=1)
Features_train, Features_test, Targetvar_train, Targetvar_test = train_test_split(Features, Targetvar, test_size=0.3, random_state=8)

#2.2 Standardise the features using StandardScaler
scaler = StandardScaler()
Features_train_scaled = scaler.fit_transform(Features_train)
Features_test_scaled = scaler.transform(Features_test)

#3 Build a Linear Regression Model
#3.1 Using Batch Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, x, y):
        m, n = x.shape
        self.theta = np.zeros(n + 1)
        x_bias = np.c_[np.ones((m,1)), x]
            
        for i in range(self.n_iterations):
            gradients = (2/m) * x_bias.T.dot(x_bias.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients
            
    def predict(self, x):
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        return x_bias.dot(self.theta)
bgd_model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
bgd_model.fit(Features_train_scaled, Targetvar_train)

#3.2 Using Stochastic Gradient Descent
sgd_model = SGDRegressor(max_iter=2000, tol=1e-5, learning_rate='constant', eta0=0.0005, random_state=8 )
sgd_model.fit(Features_train_scaled, Targetvar_train)

#4.1 Predict house values for the test set
bgd_predictions = bgd_model.predict(Features_test_scaled)
sgd_predictions = sgd_model.predict(Features_test_scaled)

#4.2 Predict house value for distric with MedInc = 8.0
newdata = Dataset[Dataset['MedInc'] == 8.0]
newfeatures = newdata.drop('MedHouseVal', axis=1)
newfeatures_scaled = scaler.transform(newfeatures)
bgd_prediction = bgd_model.predict(newfeatures_scaled)
sgd_prediction = sgd_model.predict(newfeatures_scaled)
print(f"Predicted house value (Batch Gradient Descent) for MedInc = $80,000: {bgd_prediction[0] * 100000:.2f}")
print(f"Predicted house value (Stochastic Gradient Descent) for MedInc = $80,000: {sgd_prediction[0] * 100000:.2f}")

#5.1 Plot the regression line overlaid on the test data scatter plot
plt.figure()
plt.scatter(Features_test['MedInc'], Targetvar_test, color='blue', label='Test Data', s=1)
plt.scatter(Features_test['MedInc'], bgd_predictions, color='red', label='BGD Predictions', s=1)
plt.scatter(Features_test['MedInc'], sgd_predictions, color='green', label='SGD Predictions', s=1)

#5.2 Plot label axes appropriately and include a legend
plt.title('Regression Line Over Test Data')
plt.xlabel('Median Income in 10K USD')
plt.ylabel('Median House Value in 100K USD')
plt.legend()
plt.show()

#6 Evaluate models
bgd_mse = mean_squared_error(Targetvar_test, bgd_predictions)
bgd_mae = mean_absolute_error(Targetvar_test, bgd_predictions)
bgd_r2 = r2_score(Targetvar_test, bgd_predictions)

sgd_mse = mean_squared_error(Targetvar_test, sgd_predictions)
sgd_mae = mean_absolute_error(Targetvar_test, sgd_predictions)
sgd_r2 = r2_score(Targetvar_test, sgd_predictions)

print("Batch Gradient Descent Model Evaluation:")
print(f"Mean Squared Error: {bgd_mse:.2f}")
print(f"Mean Absolute Error: {bgd_mae:.2f}")
print(f"R-squared: {bgd_r2:.2f}")

print("\nStochastic Gradient Descent Model Evaluation:")
print(f"Mean Squared Error: {sgd_mse:.2f}")
print(f"Mean Absolute Error: {sgd_mae:.2f}")
print(f"R-squared: {sgd_r2:.2f}")
