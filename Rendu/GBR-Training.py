#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# Data preapration
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# RandomizedCV Search
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV

# Models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 100)


### Data preparation

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

df = pd.read_csv('../../data/train_clean_final.csv')
df_eval = pd.read_csv('../../data/eval_clean_final.csv')

X_train, X_test, y_train, y_test = prepareData(df, True)
X_test_eval = prepareData(df_eval, False)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_eval_norm = scaler.transform(X_test_eval)

### Training with default parameters

params = {'n_estimators':500,
          'loss': 'lad',
          'verbose':1,
          'random_state':9}

reg = ensemble.GradientBoostingRegressor(**params)

reg.fit(X_train_norm, y_train)

#Train loss
y_pred_train = np.rint(reg.predict(X_train_norm))
mse_train = mean_absolute_error(y_train, y_pred_train)
#Test loss
y_pred_test = np.rint(reg.predict(X_test_norm))
mse_test = mean_absolute_error(y_test, y_pred_test)

# ------------------------------- SAVING OUTPUTS -------------------------------

#Scores
with open('results/GBR-Predictions.txt', 'w') as f:
    output = "\n========================"
    output += " Results for default Gradient Boosting Regressor (GBR) ================================"
    output += "\n With parameters :\n" + str(reg.get_params())
    output += "\n ======================================================== \n" 
    output += "MAE on TRAIN set: {:.4f}".format(mse_train) + "\n"
    output += "MAE on TEST set: {:.4f}".format(mse_test) + "\n"
    output += "\n ======================================================== \n"
    f.write(output)

#Figure
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test_norm)):
    test_score[i] = reg.loss_(y_test, y_pred)
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.savefig('results/gbr-pred-default')



# -------------------------------------------------------------------

### Training for best RandomSearchCV parameters

params = {'n_estimators': 1000,
          'max_depth': 7,
          'min_samples_split': 3,
          'learning_rate': 0.1,
          'loss': 'lad',
          'verbose':1,
          'random_state':9}

reg2 = ensemble.GradientBoostingRegressor(**params)

reg2.fit(X_train_norm, y_train)

#Train loss
y_pred_train = np.rint(reg2.predict(X_train_norm))
mse_train = mean_absolute_error(y_train, y_pred_train)
#Test loss
y_pred_test = np.rint(reg2.predict(X_test_norm))
mse_test = mean_absolute_error(y_test, y_pred_test)

# ------------------------------- SAVING OUTPUTS -------------------------------

#Scores
with open('results/GBR-Predictions.txt', 'a') as f:
    output = "\n \n ========================"
    output += " Results for optimized Gradient Boosting Regressor (GBR) =========================="
    output += "\n With parameters :\n" + str(reg2.get_params())
    output += "\n ======================================================== \n" 
    output += "MAE on TRAIN set: {:.4f}".format(mse_train) + "\n"
    output += "MAE on TEST set: {:.4f}".format(mse_test) + "\n"
    output += "\n ======================================================== \n"
    f.write(output)

#Figure
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg2.staged_predict(X_test_norm)):
    test_score[i] = reg2.loss_(y_test, y_pred)
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg2.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.savefig('results/gbr-pred-optimal')

# -------------------------------------------------------------------
