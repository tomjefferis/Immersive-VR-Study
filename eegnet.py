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


#load data from ../EEG folder, all csv files
def get_data():
    data = []
    scores = []
    order = []
    for file in [f for f in os.listdir("../EEG/") if f.endswith(".fif") and not f.endswith("resting.fif")]: # and not f.endswith("hard.fif")]:
        filepath = "../EEG/" + file
        print(file)
        # get number from file name
        order.append(int(file.split("_")[0]))
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

    return data, scores, order

def remove_participant(data, scores, order, participant):
    newdata = []
    newscores = []
    removed_participant = []
    removed_scores = []

    for x, y, z in zip(data, scores, order):
        if z != participant:
            newdata.append(x)
            newscores.append(y)
        else:
            removed_participant.append(x)
            removed_scores.append(y)
            
    return np.array(newdata), np.array(newscores), np.array(removed_participant), np.array(removed_scores)



def split_timeseries(series, window_size=1000, overlap=100):
    segments = []
    for i in range(0, series.shape[-1] - window_size + 1, window_size - overlap):
        segment = series[..., i:i + window_size]
        segments.append(segment)
    return segments

#split data into 1000 sample sliding window with 100 sample overlap
def split_data(data, scores, order, window_size=1000, overlap=100):
    X = []
    Y = []
    neworder = []
    for x, y, z in zip(data, scores, order):
        x = split_timeseries(x, window_size, overlap)
        X.extend(x)
        Y.extend([y]*len(x))
        neworder.extend([z]*len(x))
    return X, Y, neworder

def remove_nan(array,scores,order):
    mask = np.isnan(array).any(axis=(1, 2, 3))
    array = array[~mask]
    scores = scores[~mask]
    order = order[~mask]
    return array, scores, order

#window_size = 2500
#
#X, Y, order = get_data()
#X, Y = split_data(X, Y, window_size=window_size)
#
#length = len(X)
#X = np.array(X)
#X = X.reshape(length, 32, window_size, 1)
##X = np.moveaxis(X, [3], [1])
#Y = np.array(Y)
##remove nan values
#X, Y = remove_nan(X, Y)
#
#X -= np.min(X)
#X /= np.max(X)
#
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
#model = EEGNet_seq_attention(2, 32, window_size, loss='sparse_categorical_crossentropy', dropoutType='SpatialDropout2D', learning_rate=0.001)
#
#model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_split=0.1)
#
##save model
#model.save("eegnetbinary.h5")
#
##evaluate model
#model.evaluate(X_test, Y_test)
#
##plot confusion matrix 
#Y_pred = model.predict(X_test)
#Y_pred = np.argmax(Y_pred, axis=1)
#cm = confusion_matrix(Y_test, Y_pred)
#f, ax = plt.subplots(figsize=(8, 8))
#sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt= '.1f', ax=ax)
#plt.xlabel("Predicted Label")
#plt.ylabel("True Label")
#plt.title("Confusion Matrix")
#plt.show()
#
##plot loss graph
#plt.plot(model.history.history['loss'])
#plt.plot(model.history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
