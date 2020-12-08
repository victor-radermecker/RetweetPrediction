#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[31]:


### Data loading preparation

def prepareDataClassification(data, train=True):
    # data cleaning
    data.drop(columns=['text', 'hashtags', 'user_mentions', 'hashtags', 'urls', 'id'], inplace=True)
    if train:
        data['bins'] = pd.cut(data['retweet_count'], bins=[-1,1,2,3,4,10,100,1000,10000,50000,100000,200000,500000,1000000], labels=[0,1,2,3,4,10,100,1000,10000,50000,100000,200000,500000])
        X = data.drop(['classif','retweet_count'], axis=1)
        y = data['classif']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=data['bins'], test_size=0.3)
        X_train = X_train.drop(columns=['bins'])
        X_test = X_test.drop(columns=['bins'])
        return X_train, X_test, y_train, y_test
    else:
        return data

df = pd.read_csv('../../data/train_clean_final.csv')
df_eval = pd.read_csv('../../data/eval_clean_final.csv')

#Adding classification column
df['classif'] = pd.cut(df['retweet_count'], bins=[-1,4,197,1000000], labels=[0,1,2])

X_train, X_test, y_train, y_test = prepareDataClassification(df, True)
X_test_eval = prepareDataClassification(df_eval, False)

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
X_eval_norm = scaler.transform(X_test_eval)


# ### Random Forest Classifier

# In[30]:


params = {'random_state':9}


# In[29]:


rfc = RandomForestClassifier(**params, verbose=2)
rfc.fit(X_train_norm, y_train)

#Confusion matrix
y_pred=rfc.predict(X_test_norm)
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,5))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)
plt.savefig('results/Confusion_matri_RF_default')

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred, average=None))


with open('results/Classifier-results.txt', 'w') as f:
    output = "\n \n ========================"
    output += " Results for Random Forest Classifier (RFC) =========================="
    output += "\n With parameters :\n" + str(rfc.get_params())
    output += "\n ======================================================== \n" 
    output += 'Accuracy: ' + str(accuracy_score(y_test, y_pred)) + "\n"
    output += 'F1-score: ' + str(f1_score(y_test, y_pred, average=None)) + "\n"
    output += "\n ======================================================== \n"
    f.write(output)

# ### Gradient Boosting Classifier

# In[38]:


gbc = GradientBoostingClassifier(random_state=0, verbose=2)
gbc.fit(X_train_norm, y_train) 


# In[39]:


#Confusion matrix
y_pred=gbc.predict(X_test_norm)
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,5))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)
plt.savefig('results/Confusion_matri_GBC')


# In[40]:


print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred, average=None))

with open('results/Classifier-results.txt', 'a') as f:
    output = "\n \n ========================"
    output += " Results for Gradient Boosting Classifier (GBC) =========================="
    output += "\n With parameters :\n" + str(gbc.get_params())
    output += "\n ======================================================== \n" 
    output += 'Accuracy: ' + str(accuracy_score(y_test, y_pred)) + "\n"
    output += 'F1-score: ' + str(f1_score(y_test, y_pred, average=None)) + "\n"
    output += "\n ======================================================== \n"
    f.write(output)


# ### K-nearest neighbors classifier

# In[32]:


neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train_norm, y_train)  #too long


# In[33]:


#Confusion matrix
y_pred = neigh.predict(X_test_norm)
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,5))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)
plt.savefig('results/Confusion_matri_KNN')

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred, average=None))

with open('results/Classifier-results.txt', 'a') as f:
    output = "\n \n ========================"
    output += " Results for K-NN Classifier (KNN) =========================="
    output += "\n With parameters :\n" + str(neigh.get_params())
    output += "\n ======================================================== \n" 
    output += 'Accuracy: ' + str(accuracy_score(y_test, y_pred)) + "\n"
    output += 'F1-score: ' + str(f1_score(y_test, y_pred, average=None)) + "\n"
    output += "\n ======================================================== \n"
    f.write(output)


# ### Logistic regression (classifier)

lgc = LogisticRegression(random_state=9).fit(X_train_norm, y_train)

#Confusion matrix
y_pred = lgc.predict(X_test_norm)
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10,5))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)
plt.savefig('results/Confusion_matri_LGC')



print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred, average=None))


with open('results/Classifier-results.txt', 'a') as f:
    output = "\n \n ========================"
    output += " Results for K-NN Classifier (KNN) =========================="
    output += "\n With parameters :\n" + str(lgc.get_params())
    output += "\n ======================================================== \n" 
    output += 'Accuracy: ' + str(accuracy_score(y_test, y_pred)) + "\n"
    output += 'F1-score: ' + str(f1_score(y_test, y_pred, average=None)) + "\n"
    output += "\n ======================================================== \n"
    f.write(output)
