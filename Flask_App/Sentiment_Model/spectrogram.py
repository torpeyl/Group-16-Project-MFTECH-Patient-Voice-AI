from scipy import signal
import numpy as np
import wfdb
import matplotlib.pyplot as plt


record = wfdb.rdrecord('data/voice004') 


wfdb.plot_wfdb(record=record, title='Example signals')


txtFile = 'data/voice004.txt'

signalVals = np.loadtxt(txtFile)

print(signalVals)


f, t, Sxx = signal.spectrogram(signalVals)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

