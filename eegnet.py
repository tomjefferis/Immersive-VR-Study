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


# load data from ../EEG folder, all csv files
def get_data():
    data = []
    scores = []
    order = []
    filedir = '../EEG'
    for file in [f for f in os.listdir(filedir) if f.endswith(".fif") and not f.endswith("resting.fif")]:
        filepath = filedir + "/" + file
        print(file)
        # get number from file name
        order.append(int(file.split("_")[0]))
        # load raw file
        raw = mne.io.read_raw_fif(filepath, preload=True)
        # get data
        data.append(raw.get_data()[:, 7500:-15000])
        # get scores from file names, 1 = watching, 2 = normal, 3 = hard
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
        # add extra dimension for channel
        x_max = np.max(segment)
        x_avg = np.mean(segment)
        segment = (segment - x_avg) / x_max
        segment = np.expand_dims(segment, axis=-1)
        segments.append(segment)
    return segments


# split data into 1000 sample sliding window with 100 sample overlap
def split_data(data, scores, order, window_size=1000, overlap=100):
    X = []
    Y = []
    neworder = []
    for x, y, z in zip(data, scores, order):
        x = split_timeseries(x, window_size, overlap)
        X.extend(x)
        Y.extend([y] * len(x))
        neworder.extend([z] * len(x))
    return X, Y, neworder


def remove_nan(data, scores, order):
    # if series contains nan, remove it
    newdata = []
    newscores = []
    neworder = []
    for x, y, z in zip(data, scores, order):
        if np.isnan(x).any():
            continue
        else:
            newdata.append(x)
            newscores.append(y)
            neworder.append(z)
    return newdata, newscores, neworder


# def model
def model(input_shape, num_classes):
    # CNN-BILSTM
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (1, 125), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(16, (2, 1), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D((1, 4)))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


data, scores, order = get_data()
n_participants = len(set(order))
data, scores, order = split_data(data, scores, order, window_size=1250, overlap=100)
data, scores, order = remove_nan(data, scores, order)

# one hot encode scores
scores = tf.one_hot(scores, depth=3)

# use random participant as validation set
participant = np.random.randint(1, n_participants + 1)
train_data, train_scores, test_data, test_scores = remove_participant(data, scores, order, participant)

history = []

# leave one out cross validation
for i in range(n_participants):
    train_datas, train_scoress, val_data, val_scores = remove_participant(train_data, train_scores, order, i + 1)
    # model
    models = model(input_shape=(train_data.shape[1], train_data.shape[2], 1), num_classes=3)
    models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = models.fit(train_datas, train_scoress, epochs=500, batch_size=32, validation_data=(val_data, val_scores))
    history.append(hist)

# print average accuracy and loss
print("Average accuracy: ", np.mean([h.history['accuracy'] for h in history]))
print("Average loss: ", np.mean([h.history['loss'] for h in history]))

# evaluate all models on test set and save best model
for i in range(n_participants):
    models = history[i].model
    models.evaluate(test_data, test_scores)
    if i == 0:
        best_model = models
        best_acc = models.evaluate(test_data, test_scores)[1]
    else:
        if models.evaluate(test_data, test_scores)[1] > best_acc:
            best_model = models
            best_acc = models.evaluate(test_data, test_scores)[1]

# save best model
best_model.save("best_model_CNNBILSTM.h5")