#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Data Preparation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

#Metrics
from sklearn.metrics import mean_absolute_error


# ## Loading and preparing dataset


def prepareData(data, train=True):
    # data cleaning
    data.drop(columns=['text', 'hashtags', 'user_mentions', 'hashtags', 'urls', 'id'], inplace=True)
    if train:
        data['bins'] = pd.cut(data['retweet_count'], bins=[-1,1,2,3,4,10,100,1000,10000,50000,100000,200000,500000,1000000], labels=[0,1,2,3,4,10,100,1000,10000,50000,100000,200000,500000])
        X = data.drop('retweet_count', axis=1)
        y = data['retweet_count']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=data['bins'], test_size=0.3)
        X_train = X_train.drop(columns=['bins'])
        X_test = X_test.drop(columns=['bins'])
        return X_train, X_test, y_train, y_test
    else:
        return data


# In[8]:
df = pd.read_csv('../../data/train_clean_final.csv')
df_eval = pd.read_csv('../../data/eval_clean_final.csv')

X_train, X_test, y_train, y_test = prepareData(df, True)
X_test_eval = prepareData(df_eval, False)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_eval_norm = scaler.transform(X_test_eval)

results = []

# ## Testing different regressors without tunning

# Global metric for all ours algorithms
def metrics(Xtrain, Xtest, ytrain, ytest, model, model_name):
    #Train loss
    y_pred_train = np.rint(model.predict(Xtrain))
    mse_train = mean_absolute_error(ytrain, y_pred_train)
    #Test loss
    y_pred_test = np.rint(model.predict(Xtest))
    mse_test = mean_absolute_error(ytest, y_pred_test)

    output = "\n========================"
    output += f' Results for {model_name}'
    output += "========================" 
    output += "\n MAE on TRAIN set: {:.4f}".format(mse_train) 
    output += "\n MAE on TEST set: {:.4f}".format(mse_test)
    output += "\n ================================================ \n "
    return output


# #### Gradient Boosting Regressor
gbr = GradientBoostingRegressor(loss='lad')
gbr.fit(X_train_norm, y_train)
result_gbr = metrics(X_train_norm, X_test_norm, y_train, y_test, gbr, "Gradient Boosting Regressor")
print(result_gbr)
results.append(result_gbr)

# #### Random Forest Regressor
rfr = RandomForestRegressor(verbose=2)
rfr.fit(X_train_norm, y_train)
result_rfr = metrics(X_train_norm, X_test_norm, y_train, y_test, rfr, "Random Forest Regressor")
print(result_rfr)
results.append(result_rfr)

# #### Logistic regression
rfr = RandomForestRegressor(verbose=2)
rfr.fit(X_train_norm, y_train)
result_rfr = metrics(X_train_norm, X_test_norm, y_train, y_test, rfr, "Random Forest Regressor")
print(result_rfr)
results.append(result_rfr)


#Store results in text file
with open('results/regression-comparison', 'w') as f:
    for res in results:
        f.write(res)





