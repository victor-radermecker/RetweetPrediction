import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
import csv

'''Network Definition'''
def build_and_compile_model(n_neurons):
  model = keras.Sequential([
      layers.Dense(n_neurons,input_shape=(9,) ),
      layers.Dense(n_neurons, activation='relu'),
      layers.Dense(n_neurons, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

'''Function to plot error during the training'''
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  

'''Import and split dataset'''
dataset = pd.read_csv('export_dataframe.csv')
train_features, test_features, train_labels, test_labels = scsplit(dataset, dataset['retweet_count'],stratify=dataset['retweet_count'], train_size=0.7, test_size=0.3)
del train_features['retweet_count'], test_features['retweet_count']

'''Compile model'''
RTmodel = build_and_compile_model(64)
#RTmodel = keras.models.load_model('RTmodel') #Chargement du modèle si préexistant
RTmodel.summary()

'''Train model'''
history = RTmodel.fit(
    train_features, train_labels,
      validation_data = (test_features, test_labels), #validation_split=0.2,
    verbose=1, epochs=100)
RTmodel.save('RTmodel')
plot_loss(history)

'''Prediction on test dataset'''
y_pred = RTmodel.predict(test_features)
print("MAE :")
print(mean_absolute_error(y_true=test_labels, y_pred=y_pred))


'''Use of the NN on the evaluation dataset'''
eval_data = pd.read_csv('evaluation_processed.csv')
eval_data_id = eval_data['id'].copy(deep=True)
del eval_data['id']

y_pred_test = RTmodel.predict(eval_data)

eval_data['id'] = eval_data_id.copy(deep=True)

with open("gbr_predictions_NN.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "NoRetweets"])
    for index, prediction in enumerate(y_pred_test):
        writer.writerow([str(eval_data['id'].iloc[index]) , str(int(prediction))])