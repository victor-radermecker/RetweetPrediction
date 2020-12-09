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
from sklearn.ensemble import RandomForestClassifier

# Models
from sklearn.ensemble import GradientBoostingRegressor


# --------------- DATA PREPARATION -------------------
def prepareData(data, train=True):
    # data cleaning
    data.drop(columns=['text', 'hashtags', 'user_mentions', 'hashtags', 'urls', 'id'], inplace=True)
    if train:
        data['bins'] = pd.cut(data['retweet_count'], bins=[-1,1,2,3,4,10,100,1000,10000,50000,100000,200000,500000,1000000], labels=[0,1,2,3,4,10,100,1000,10000,50000,100000,200000,500000])
        data = data.sample(50000)
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
X_train, X_test, y_train, y_test = prepareData(df)
X_test_eval = prepareData(df_eval, False)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_eval_norm = scaler.transform(X_test_eval)


# --------------- GRADIENT BOOSTING OPTIMIZATION -------------------

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



rf = RandomForestClassifier(random_state = 9)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=9, n_jobs = -1)
rf_random.fit(X_train_norm, y_train)


with open('results/RF-tuning.txt', 'w') as f:
    output = "\n========================"
    output += " Results from Random Search "
    output += "=========================" 
    output += "\n The best estimator across ALL searched params:\n" + str(randm.best_estimator_)
    output += "\n The best score across ALL searched params:\n" + str(randm.best_score_)
    output += "\n The best parameters across ALL searched params:\n" + str(randm.best_params_)
    output += "\n ========================================================"
    f.write(output)