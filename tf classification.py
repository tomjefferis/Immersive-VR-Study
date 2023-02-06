import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import scipy.io

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
                scores.append(1)
            elif "normal" in file or "correct" in file:
                scores.append(2)
            elif "hard" in file:
                scores.append(3)

    return data, scores


[data,scores] = get_data()
def flatten(l):
    return [item for sublist in l for item in sublist]

def split_timeseries(series, window_size=1000, overlap=100):
    segments = []
    for i in range(0, series.shape[-1] - window_size + 1, window_size - overlap):
        segment = series[..., i:i + window_size]
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

def normalize(X):
    mean = np.mean(X, axis=2)
    std = np.std(X, axis=2)
    X = (X - mean[None, :, :, None]) / std[None, :, :, None]
    return X

#normalize data
for i in range(len(X)):
    X[i] = normalize(X[i])

#reshape data
length = len(X)
X = np.array(X)
X = X.reshape(length, 32, 31, 1000)
Y = np.array(Y)
#split data into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

n_classes = 2

# Define the input layer
inputs = tf.keras.layers.Input(shape=(32, 31, 1000))

# Add 2D Convolutional layer
conv2d_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(32, 1), strides=(1, 1), activation='relu')(inputs)

# Add another 2D Convolutional layer
conv2d_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 20), strides=(1, 1), activation='relu')(conv2d_1)

# Add 2D Max Pooling layer
max_pool = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1))(conv2d_2)

# Flatten the output of the 2D Max Pooling layer
flatten = tf.keras.layers.Flatten()(max_pool)

# Add a fully connected layer with ReLU activation
fc = tf.keras.layers.Dense(units=128, activation='relu')(flatten)

# Add the final output layer with softmax activation
outputs = tf.keras.layers.Dense(units=n_classes, activation='softmax')(fc)

# Define the model
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=500, batch_size=64)

# Evaluate the model
model.evaluate(X_test, Y_test)

# Save the model
model.save('model.h5')

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



