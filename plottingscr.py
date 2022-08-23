#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import pandas as pd

#load data
data = pd.read_csv('EEGCh3_D3357722440D_13.47.33_250.csv')

# Number of samplepoints
N = len(data.iloc[:,0])
# sample spacing
T = data.values[0,0]
#x = first column of data
x = data.values[:,0]
#y = second column of data
y = data.values[:,1]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()
plt.xlim[0,20]