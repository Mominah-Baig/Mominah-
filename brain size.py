# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:34:36 2020

@author: Mominah Baig
"""

#simple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset of brain weight and head size
dataset = pd.read_csv('dataset.csv')
A = dataset.iloc[:, 2:3].values
B = dataset.iloc[:, 3].values
#splitting the dataset into training and test set with the ratio of 4:1
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A,B, test_size=0.2)
#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#to train the model
regressor.fit(A_train,B_train)
#predicting the test set results
B_pred=regressor.predict(A_test)
#plotting training set
plt.scatter(A_train, B_train, color='blue')
plt.plot(A_train, regressor.predict(A_train), color='orange')
plt.title('Head Size(cm^3) Vs. Brain Weight(grams)')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.show()
#plotting Test set results
plt.scatter(A_test, B_test, color= 'blue')
plt.plot(A_train, regressor.predict(A_train), color='orange')
plt.title('Head Size(cm^3) Vs. Brain Weight(grams)')
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.show()