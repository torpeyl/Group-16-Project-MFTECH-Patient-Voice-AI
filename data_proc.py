# from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

import os

import wfdb
directory = 'data'

# define parameters
SAMPLING_FREQUENCY = 8000   # Hz
T_PERIOD = 1/SAMPLING_FREQUENCY
N = SAMPLING_FREQUENCY*5
GAUSSIAN_STD = 8  # standard deviation for Gaussian window in samples


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


def init_SFT():
    t_x = np.arange(N) * T_PERIOD  # time indexes
    f_i = 5e-3*(t_x - t_x[N // 3])**2 + 1  # sample frequencies

    window = gaussian(50, std=GAUSSIAN_STD, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(window, hop=10, fs=SAMPLING_FREQUENCY, mfft=200)
    return t_x, f_i, SFT


def get_spectrogram(SFT:ShortTimeFFT, filename:str):
    x = np.squeeze(get_audio_array(filename))
    Sx = SFT.spectrogram(x)     # spectrogram of x
    return Sx


def plot_spectrogram(t_x:np.ndarray, f_i:np.ndarray, Sx:np.ndarray, SFT:ShortTimeFFT, sample_label:str=""):
    fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    ax1.set_title(rf"Spectrogram {sample_label} ({SFT.m_num*SFT.T:g}$\,s$ Gaussian " +
                rf"window, $\sigma_t={GAUSSIAN_STD*SFT.T:g}\,$s)")
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
                rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
                rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
            xlim=(t_lo, t_hi))
    Sx_dB = 10 * np.log10(np.fmax(Sx, 1e-4))  # limit range to -40 dB
    im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto',
                    extent=SFT.extent(N), cmap='magma')
    ax1.plot(t_x, f_i, 'g--', alpha=.5, label='$f_i(t)$')
    fig1.colorbar(im1, label='Power Spectral Density ' +
                            r"$20\,\log_{10}|S_x(t, f)|$ in dB")
    plt.show()


if __name__ == "__main__":
    filename = "voice001"

    t_x, f_i, SFT = init_SFT()
    Sx = get_spectrogram(SFT, filename)
    plot_spectrogram(t_x, f_i, Sx, SFT, filename)

