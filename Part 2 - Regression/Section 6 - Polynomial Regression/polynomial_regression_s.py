# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:45:09 2017

@author: mir
"""

# Polynomial regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #make shure X is a matrix
y = dataset.iloc[:, 2].values

# Fitting linear regression to the dataset
# Fitting multiple linear regression to the trainingset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynimial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visuallising the linear regression model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Level vs Salery')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visuallising the polynmial regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Level vs Salery (Polynomial)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predicting the result with linear regression
lin_reg.predict(6.5)

# predicting the result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))