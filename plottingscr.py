#matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from mne_icalabel import label_components

#load data
data = pd.read_csv('EEGCh1_D3357722440D_13.47.33_250.csv')


#data = pd.read_csv('data.csv',index_col=0)
#data.index = pd.to_datetime(data.index)
#data = data['max_open_files'].astype(float).values

N = data.shape[0] #number of elements
t = data.iloc[:,0] #converting hours to seconds
s = data.iloc[:,1]

fft = np.fft.fft(s)
fftfreq = np.fft.fftfreq(len(s))

T = t[1] - t[0]

f = np.linspace(0, 1 / T, N)
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.plot(fftfreq,fft)
plt.show()