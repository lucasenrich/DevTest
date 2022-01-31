# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:17:10 2022

@author: HP
"""
from joblib import dump, load
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
dump(reg, 'reg_model_1.joblib') 