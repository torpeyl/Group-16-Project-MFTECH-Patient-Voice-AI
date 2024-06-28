import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io.wavfile as wavfile
import os
matplotlib.use('Agg')

def wav_to_spec(wav_file_path):
    Fs, aud = wavfile.read(wav_file_path)
    if aud.size > 25000:
        print(f"Received .wav file path.")
        #Limit audio range from 0.1 to .5 seconds, i.e. each clip becomes 0.4s long
        lower_bound = 5000
        upper_bound = 25000
        upper_bound = aud.size - upper_bound

        aud = aud[lower_bound:-upper_bound]
        #print(aud.size)
        #fig = plt.figure(dpi=600)  # Set the desired DPI here
        # Generate a spectrogram
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(aud, Fs=Fs)
        print(f"Completed conversion to spectrogram.")
        plt.axis('off')
        # Display the plot{{
        spectro_file_path = './downloads/spectrogram.jpg'
        plt.savefig(f'{spectro_file_path}', dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Spectrogram saved as .jpg.")
        return f'{spectro_file_path}'
