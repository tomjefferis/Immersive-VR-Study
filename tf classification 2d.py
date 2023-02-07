import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import scipy.io
from tensorflow.keras import layers

#load data from ../EEG folder, all csv files
def get_data():
    data = []
    scores = []
    for file in os.listdir("../FreqData/"):
        if file.endswith(".mat") and not file.endswith("resting.mat") and not file.endswith("hard.mat"):
            filepath = "../FreqData/" + file
            print(file)
            #load mat file
            raw = scipy.io.loadmat(filepath)
            data.append(raw['temp'])
            #get scores from file names, 1 = watching, 2 = normal, 3 = hard
            if "watching" in file or "watch" in file:
                scores.append(0)
            elif "normal" in file or "correct" in file:
                scores.append(1)
            elif "hard" in file:
                scores.append(2)

    return data, scores


[data,scores] = get_data()
def flatten(l):
    return [item for sublist in l for item in sublist]

def split_timeseries(series, window_size=10000, overlap=100):
    segments = []
    for i in range(0, series.shape[-1] - window_size + 1, window_size - overlap):
        segment = series[..., i:i + window_size]
        segment = np.average(segment, axis=2)
        segments.append(segment)
    return segments

#split data into 300 sample sliding window with 100 sample overlap
def split_data(data, scores):
    X = []
    Y = []
    for i in range(len(data)):
        x = split_timeseries(data[i])
        X.append(x)
        y = np.full((len(x),1), scores[i])
        Y.append(y)
    return flatten(X), flatten(Y)


X, Y = split_data(data, scores)

def remove_nan(array,scores):
    mask = np.isnan(array).any(axis=(1, 2, 3))
    array = array[~mask]
    scores = scores[~mask]
    return array, scores

#reshape data
length = len(X)
X = np.array(X)
X = X.reshape(length, 32, 31, 1)
#X = np.moveaxis(X, [3], [1])

Y = np.array(Y)
#remove nan values
X, Y = remove_nan(X, Y)

#X -= np.min(X)
#X /= np.max(X)

#split data into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

input_shape = (32, 31, 1)

model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y_train, epochs=600, batch_size=8, validation_split=0.1)
# Save the model
model.save('model2d.h5')
# Evaluate the model
model.evaluate(X_test, Y_test)

#plot confusion matrix and accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = Y_test
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt= '.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("Accuracy: ", accuracy_score(Y_true, Y_pred_classes))



