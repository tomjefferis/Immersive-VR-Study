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
from sklearn.preprocessing import OneHotEncoder

#load data from ../EEG folder, all csv files
def get_data():
    data = []
    scores = []
    order = []
    for file in [f for f in os.listdir("../EEG/") if f.endswith(".fif") and not f.endswith("resting.fif")]:
        filepath = "../EEG/" + file
        print(file)
        # get number from file name
        order.append(int(file.split("_")[0]))
        #load raw file
        raw = mne.io.read_raw_fif(filepath, preload=True)
        #get data from fp1 & fp2
        data.append(raw.get_data(picks=["Fp1", "Fp2"])[:, 7500:-15000])
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
        x_max = np.max(segment)
        x_avg = np.mean(segment)
        segment = (segment - x_avg) / x_max
        # add extra dimension for channel
        segment = np.expand_dims(segment, axis=-1)
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

def remove_nan(data,scores,order):
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

data, scores, order = get_data()
n_participants = len(set(order))
data, scores, order = split_data(data, scores, order, window_size=2500, overlap=250)
data, scores, order = remove_nan(data, scores, order)

# one hot encode scores
enc = OneHotEncoder(handle_unknown='ignore')
scores = enc.fit_transform(np.array(scores).reshape(-1, 1)).toarray()


# use random participant as validation set
participant = np.random.randint(1, n_participants+1)
train_data, train_scores, test_data, test_scores = remove_participant(data, scores, order, participant)

history = []

# leave one out cross validation
for i in range(n_participants):
    train_data, train_scores, val_data, val_scores = remove_participant(train_data, train_scores, order, i+1)
    # model
    model= EEGNet_seq(3, Chans=2, Samples=2500, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(train_data, train_scores, epochs=500, batch_size=32, validation_data=(val_data, val_scores))
    history.append(hist)
    

# print average accuracy and loss
print("Average accuracy: ", np.mean([h.history['accuracy'] for h in history]))
print("Average loss: ", np.mean([h.history['loss'] for h in history]))

# save the best model 
for h in history:
    if h.history['val_accuracy'] == np.max([h.history['val_accuracy'] for h in history]):
        best_model = h.model
        break

# save best model
best_model.save("best_model_CNNBILSTM.h5")







