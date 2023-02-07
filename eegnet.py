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
from classifier_gen import EEGNet_seq


#load data from ../EEG folder, all csv files
def get_data():
    data = []
    scores = []
    for file in [f for f in os.listdir("../EEG/") if f.endswith(".fif") and not f.endswith("resting.fif")]: # and not f.endswith("hard.fif")]:
        filepath = "../EEG/" + file
        print(file)
        #load raw file
        raw = mne.io.read_raw_fif(filepath, preload=True)
        #get data
        data.append(raw.get_data()[:,7500:-15000])        
        #get scores from file names, 1 = watching, 2 = normal, 3 = hard
        if "watching" in file or "watch" in file:
            scores.append(0)
        elif "normal" in file or "correct" in file:
            scores.append(1)
        elif "hard" in file:
            scores.append(2)

    return data, scores

def flatten(l):
    return [item for sublist in l for item in sublist]

def split_timeseries(series, window_size=1000, overlap=100):
    segments = []
    for i in range(0, series.shape[-1] - window_size + 1, window_size - overlap):
        segment = series[..., i:i + window_size]
        segments.append(segment)
    return segments

#split data into 1000 sample sliding window with 100 sample overlap
def split_data(data, scores, window_size=1000, overlap=100):
    X = []
    Y = []
    for x, y in zip(data, scores):
        x = split_timeseries(x, window_size, overlap)
        X.extend(x)
        Y.extend([y]*len(x))
    return X, Y

def remove_nan(array,scores):
    mask = np.isnan(array).any(axis=(1, 2, 3))
    array = array[~mask]
    scores = scores[~mask]
    return array, scores

window_size = 2500

X, Y = get_data()
X, Y = split_data(X, Y, window_size=window_size)

length = len(X)
X = np.array(X)
X = X.reshape(length, 32, window_size, 1)
#X = np.moveaxis(X, [3], [1])
Y = np.array(Y)
#remove nan values
X, Y = remove_nan(X, Y)

X -= np.min(X)
X /= np.max(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = EEGNet_seq(3, 32, window_size, loss='sparse_categorical_crossentropy', dropoutType='SpatialDropout2D', learning_rate=0.001)

model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.1)

#save model
model.save("eegnet.h5")

#evaluate model
model.evaluate(X_test, Y_test)

#plot confusion matrix 
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(Y_test, Y_pred)
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt= '.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#plot loss graph
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
