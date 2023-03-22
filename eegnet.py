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
    for file in [f for f in os.listdir("../EEG/") if f.endswith(".fif") and not f.endswith("resting.fif") and not f.endswith("hard.fif")]:
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

