import os
import pandas as pd
import mne
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
from classifier_gen import EEGNet_seq
from sklearn.model_selection import train_test_split
import numpy as np
#load data from ../EEG folder, all csv files
def get_data():
    data = []
    scores = []
    for file in os.listdir("../EEG/"):
        if file.endswith(".fif") and not file.endswith("resting.fif"):
            filepath = "../EEG/" + file
            print(file)
            raw = mne.io.read_raw_fif(filepath).load_data().get_data()
            raw = raw[:,0:150000]
            #if raw is less than 150000, pad with zeros
            if raw.shape[1] < 150000:
                raw = pad_sequences(raw, maxlen=150000, dtype='float32', padding='post', truncating='post', value=0)
            data.append(raw)
            #get scores from file names, 1 = watching, 2 = normal, 3 = hard
            if "watching" in file or "watch" in file:
                scores.append(1)
            elif "normal" in file or "correct" in file:
                scores.append(2)
            elif "hard" in file:
                scores.append(3)

    return data, scores


[data,scores] = get_data()

#for each data file, get the first 150000 samples
#datas = [x[:, 0:150000] for x in data]
chan, samples = data[0].shape
length = len(data)
#reshape data to be 3d
data = np.reshape(data, (30, 31, 150000,1))
#sklearn test train split
X_train, X_test, y_train, y_test = train_test_split(data, np.asarray(scores), test_size=0.2, random_state=42)

#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


BATCH_SIZE = 10
SHUFFLE_BUFFER_SIZE = 100

#train_dataset = train_dataset.batch(BATCH_SIZE)
#test_dataset = test_dataset.batch(BATCH_SIZE)
#get data chan and samples


model = EEGNet_seq(3,chan,samples)
model.fit(X_train,y_train, epochs=1000,batch_size=10)
model.evaluate(X_test,y_test)
