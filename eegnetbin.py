import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import scipy.io
from tensorflow.keras import layers
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from classifier_gen import EEGNet_seq, EEGNet_seq_attention
from eegnet import get_data, split_data, remove_nan, remove_participant, split_timeseries

window_size = 2500

X, Y, order = get_data()
n_participants = len(X)
X, Y, order = split_data(X, Y, order, window_size=window_size)

length = len(X)
X = np.array(X)
X = X.reshape(length, 32, window_size, 1)
#X = np.moveaxis(X, [3], [1])
Y = np.array(Y)
order = np.array(order)
#remove nan values
X, Y, order = remove_nan(X, Y, order)
X -= np.min(X)
X /= np.max(X)

history = []

for i in range(n_participants):
    X_train, Y_train, X_test, Y_test = remove_participant(X, Y, order, i)
    model = EEGNet_seq(2, 32, window_size, loss='sparse_categorical_crossentropy', dropoutType='SpatialDropout2D', learning_rate=0.001)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history[i] = model.fit(X_train, Y_train, epochs=200, validation_split=0.1)

