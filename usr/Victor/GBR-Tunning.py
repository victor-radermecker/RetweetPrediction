import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

#Data Preparation
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# RandomizedCV Search
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RandomizedSearchCV

# Models
from sklearn.ensemble import GradientBoostingRegressor


# --------------- DATA PREPARATION -------------------
def prepareData(data, train=True):
    # data cleaning
    data.drop(columns=['text', 'hashtags', 'user_mentions', 'hashtags', 'urls', 'id'], inplace=True)
    if train:
        X = data.drop('retweet_count', axis=1)
        y = data['retweet_count'].to_numpy()
        return train_test_split(X, y, test_size=0.2)
    else:
        return data
		
df = pd.read_csv('../../data/train_clean_final.csv')
df_eval = pd.read_csv('../../data/eval_clean_final.csv')
X_train, X_test, y_train, y_test = prepareData(df)
X_test_eval = prepareData(df_eval, False)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_eval_norm = scaler.transform(X_test_eval)


# --------------- GRADIENT BOOSTING OPTIMIZATION -------------------

params = {'n_estimators': 1000,
          'max_depth': 7,
          'min_samples_split': 3,
          'learning_rate': 0.01,
          'loss': 'lad',
           'verbose':1}

model = GradientBoostingRegressor(loss='lad')

parameters = {'learning_rate': sp_randFloat(),
              'subsample'    : sp_randFloat(),
              #'n_estimators' : sp_randInt(100, 1000),
              'max_depth'    : sp_randInt(4, 10)}

randm = RandomizedSearchCV(estimator=model, 
                           param_distributions = parameters, 
                           scoring = 'neg_mean_absolute_error',
                           cv = 2, 
                           n_iter = 20, 
                           n_jobs=-1,
                           verbose=1)

randm.fit(X_train_norm, y_train)

with open('results/optimization.txt', 'w') as f:
    output = "\n========================================================"
    output += " Results from Random Search "
    output += "========================================================" 
    output += "\n The best estimator across ALL searched params:\n" + str(randm.best_estimator_)
    output += "\n The best score across ALL searched params:\n" + str(randm.best_score_)
    output += "\n The best parameters across ALL searched params:\n" + str(randm.best_params_)
    output += "\n ========================================================"
    f.write(output)