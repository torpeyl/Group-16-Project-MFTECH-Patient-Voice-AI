import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
import librosa

import os

import wfdb
directory = 'data'


def get_audio_array(filename:str):
    f = os.path.join(directory, filename)
    record = wfdb.rdrecord(f) 
    record_attributes = record.__dict__
    audio = np.asarray(record_attributes["p_signal"])
    return audio

def plot_audio(filename:str):
    f = os.path.join(directory, filename)
    record = wfdb.rdrecord(f)
    wfdb.plot_wfdb(record=record, title=filename)


class stft:
    def __init__(self):
        self.f_s = 8000   # sampling frequency (in Hz)
        self.t_period = 1/self.f_s
        self.n = self.f_s*5
        self.std = 8  # standard deviation for Gaussian window in samples

        self.t_x = np.arange(self.n) * self.t_period  # array of time indexes
        self.f_i = 5e-3*(self.t_x - self.t_x[self.n // 3])**2 + 1  # array of sample frequencies

        self.window = gaussian(50, std=self.std, sym=True)  # symmetric Gaussian window
        self.SFT = ShortTimeFFT(self.window, hop=10, fs=self.f_s, mfft=200)
        
    def get_spectrogram(self, signal):
        Sx = self.SFT.spectrogram(signal)     # spectrogram of x
        return Sx
    
    def plot_spectrogram(self, signal, label):
        Sx = self.get_spectrogram(signal)

        fig1, ax1 = plt.subplots(figsize=(7., 4.))  # enlarge plot a bit
        t_lo, t_hi = self.SFT.extent(self.n)[:2]  # time range of plot
        ax1.set_title(rf"Spectrogram {label} ({self.SFT.m_num*self.SFT.T:g}$\,s$ Gaussian " +
                    rf"window, $\sigma_t={self.std*self.SFT.T:g}\,$s)")
        ax1.set(xlabel=f"Time $t$ in seconds ({self.SFT.p_num(self.n)} slices, " +
                    rf"$\Delta t = {self.SFT.delta_t:g}\,$s)",
                ylabel=f"Freq. $f$ in Hz ({self.SFT.f_pts} bins, " +
                    rf"$\Delta f = {self.SFT.delta_f:g}\,$Hz)",
                xlim=(t_lo, t_hi))
        Sx_dB = 10 * np.log10(np.fmax(Sx, 1e-4))  # limit range to -40 dB
        im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto',
                        extent=self.SFT.extent(self.n), cmap='magma')
        ax1.plot(self.t_x, self.f_i, 'g--', alpha=.5, label='$f_i(t)$')
        fig1.colorbar(im1, label='Power Spectral Density ' +
                                r"$20\,\log_{10}|S_x(t, f)|$ in dB")
        
        plt.savefig(rf"img/{label}.png")


if __name__ == "__main__":
    filename = "voice003"
    x = np.squeeze(get_audio_array(filename))

    # filename2 = "voice100"
    # x2 = np.squeeze(get_audio_array(filename2))

    # SFT = stft()
    # SFT.plot_spectrogram(x,filename)
    # SFT.plot_spectrogram(x2,filename2)

    signal = x
    sr = 8000   # sampling rate
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=20, sr=sr)
    print(mfccs.shape)
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mfccs, 
                            x_axis="time", 
                            sr=sr)
    plt.colorbar(format="%+2.f")
    plt.savefig(filename+".png")

