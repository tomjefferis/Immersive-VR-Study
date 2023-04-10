import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import scipy.io
from tensorflow.keras import layers
import mne
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

#load data from ../EEG folder, all csv files
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
        data.append(raw.get_data(picks=['Fp1','Fp2'])[:, 7500:-15000])
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
    model.add(layers.Conv2D(32, (2, 1), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(16, (2, 1), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu'))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


window_size = 2500
channels = 2


data, scores, order = get_data()
n_participants = len(set(order))
data, scores, order = split_data(data, scores, order, window_size=window_size, overlap=100)
#data, scores, order = remove_nan(data, scores, order)

#scored = scores
# one hot encode scores sklearn
scores = preprocessing.OneHotEncoder().fit_transform(np.array(scores).reshape(-1, 1))
scores = scores.toarray()

# use test train split inc order
train_data, test_data, train_scores, test_scores, train_order, test_order = train_test_split(data, scores, order,test_size=0.2, random_state=42, shuffle=True)
# test data into array
test_data = np.array(test_data)
#test_scores = test_scores.tolist()
history = []
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

# define the checkpoint path and filename
checkpoint_path = "best_model_CNNLSTM.h5"

# define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='min')



# leave one out cross validation
for i in range(n_participants):
    train_datas, train_scoress, val_data, val_scores = remove_participant(train_data, train_scores, train_order, i + 1)
    # model
    models = model((channels, window_size, 1), num_classes=3)
    models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = models.fit(train_datas, train_scoress, epochs=500, batch_size=128, validation_data=(val_data, val_scores),callbacks=[early_stop, checkpoint])
    hist = load_model(checkpoint_path)
    history.append(hist)


# evaluate all models on test set and save best model
best_model = None
best_acc = 0
# change test_scores to list instead of array


for i, model in enumerate(history):
    # evaluate model on test set
    _, acc = model.evaluate(test_data, test_scores, verbose=0)
    print('Model %d: %.3f' % (i + 1, acc))
    # check if best model
    if acc > best_acc:
        best_acc = acc
        best_model = model

# save best model
best_model.save("best_model_CNNBILSTM.h5")
