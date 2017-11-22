# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:50:31 2017

@author: mir
"""

# Multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting multiple linear regression to the trainingset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the testset results
y_pred = regressor.predict(X_test)

# Building the optimal model using backward elimination
import statsmodels.formula.api as sm
# Adding a collum of once for the constant b0
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) 
# Step 2:initialize the optimal set of variables and include all collums
# and fit the full modell with all possible predictors
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3: Consider the predictor with the highast p-value (Possibility value, the lower, the more significant is the realated independent variable)
regressor_OLS.summary()
# Step 5: 1. itteration
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3: Consider the predictor with the highast p-value (Possibility value, the lower, the more significant is the realated independent variable)
regressor_OLS.summary()
# Step 5: 2. itteration
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3: Consider the predictor with the highast p-value (Possibility value, the lower, the more significant is the realated independent variable)
regressor_OLS.summary()
# Step 5: 3. itteration
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3: Consider the predictor with the highast p-value (Possibility value, the lower, the more significant is the realated independent variable)
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3: Consider the predictor with the highast p-value (Possibility value, the lower, the more significant is the realated independent variable)
regressor_OLS.summary()
