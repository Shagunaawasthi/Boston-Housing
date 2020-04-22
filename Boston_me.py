# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:48:37 2020

@author: shaguna awasthi
"""


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
boston = pd.read_csv("Boston.csv")
#structure of data and to see if data file is loaded properly
str(boston)
#displaying column names to find the target column name

boston.columns
#target column
y=boston.medv
print(y)
#displaying correlation
corr=boston.corr()
#from corr we can clearly see that lsat,ptratio has a strong negative correlation with the target
#and rm has a strong positive correlation
#feature matrix
feature=['rm','lstat','ptratio']
X=boston[feature]

#visualizing the feature matrix so as to find any abnormality
#like negative values etc
X.describe()
X.head()
##MODEL 1
#splitting data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
##splitting train into validation set, 0.25*0.8=0.2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
##performing linear regression
boston_model=LinearRegression()
#fitting data on train set
boston_model.fit(X_train,y_train)
#making predictions
y_train_pred=boston_model.predict(X_train)
print(y_train_pred)
##printing first five values of predicted and orignal observations 
#for rough comparison
print(boston_model.predict(X_train.head()))
print(y_train.head())
#printing r squared(0.6587)
r2 = r2_score(y_train, y_train_pred)
print(r2)
#printing r squared(0.6433)
n =len(X_train)
r2_adj =1- (1-r2)*(n-1)/(n-(13+1))
print(r2_adj)

##for model summary
X_train1 = sm.add_constant(X_train)
model =sm.OLS(y_train,X_train1).fit()
##printing model coefficients
print(model.params)
##model summary
print(model.summary())
#root mean square if rmse(5.031)
rmse_train=np.sqrt(mean_squared_error(y_train,y_train_pred))
print(rmse_train)


##WAS TRYING TO SCALE BY TAKING LOG. R AND ADJUSTED R SQUARE CAME MORE THAN MODEL1
##BUT RMSE CAME LESS DIDN'T HAD TIME TO FULLY IMPLEMENT IT


##MODEL 2
##for scaling purposes taking log of target value i.e y
#y_log_train = np.log(y_train)
#print(y_train)
#print(y_log_train.head())
##model with log of target

#boston_model2=LinearRegression()
#boston_model2.fit(X_train,y_log_train)
#y_log_train_pred=boston_model.predict(X_train)
#print(y_train_pred)

#model2=sm.OLS(y_log_train,X_train1).fit()
#print(model2.summary())

##COMPARING model and log model
##non log
#Bmodel2=boston_model.fit(X_train1,y_train)
#r22=r2_score(y_train,Bmodel2.predict(X_train1))
#print(r22)

##log
#Bmodel3=boston_model.fit(X_train1,y_log_train)
#r23=r2_score(y_log_train,Bmodel3.predict(X_train1))
#r23=Bmodel3.score(X_train1,y_log_train)
#print(r23)
#y_pred_log=np.exp(Bmodel3.predict(X_train1))
#print(y_pred_log)
#rmse_train_log=np.sqrt(mean_squared_error(y_log_train, y_pred_log))
#print(rmse_train_log)
##  rmse for nonlog=5.264 and for log scaled model rmse=0.225
##




##residual plot
Bmodel2 =boston_model.fit(X_train1, y_train)
pred_train = Bmodel2.predict(X_train1)
plt.scatter(pred_train, y_train - pred_train)
plt.xlabel("Predicted Y")
plt.ylabel("Residual")
plt.title("Residual Plot using Training data")
##Since the residual plot achived is scattered and no pattern is found
##linearity property should be satisfied


##evaluatinhg on validation set
Bvalmodel=boston_model.fit(X_train,y_train)
y_val_pred=Bvalmodel.predict(X_val)
print(y_val_pred)
print(y_val)
#rsquare(0.72494)
r2f = r2_score(y_val, y_val_pred)
print(r2f)
#adjusted r square(0.6838)
n =len(X_val)
r2_adj_f =1- (1-r2f)*(n-1)/(n-(13+1))
print(r2_adj_f)

#root mean square
rmse_val=np.sqrt(mean_squared_error(y_val,y_val_pred))
print(rmse_val)
##testing on test set
Btestmodel=boston_model.fit(X_train,y_train)
y_test_pred=Btestmodel.predict(X_test)
print(y_test_pred)
print(y_test)
## r square(0.5838)
r2t = r2_score(y_test, y_test_pred)
print(r2t)
##adjusted r square (0.5223)
n =len(X_test)
r2_adj_t =1- (1-r2t)*(n-1)/(n-(13+1))
print(r2_adj_t)
##rmse(5.517)
rmse_test=np.sqrt(mean_squared_error(y_test,y_test_pred))
print(rmse_test)

#y_pred=boston_model.predict(X_test)
#print(boston_model.predict(X_test.head()))
#print(y_test.head())

## Visualizing the differences between actual prices 
#and predicted values
plt.scatter( y_test, y_test_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Plot")
plt.show()

## Checking residuals
plt.scatter(y_test,y_test-y_test_pred)
plt.title("Predicted vs residuals")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

##There is no particular pattern 
##plot and values are distributed equally around zero. 
##Linearity assumption is satisfied


##  i googled how to draw histograms 
##to check normal distribution of errors
# Checking Normality of errors
sns.distplot(y_test-y_test_pred)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

## the residuals are normally distributed. 
##So normality assumption is satisfied


