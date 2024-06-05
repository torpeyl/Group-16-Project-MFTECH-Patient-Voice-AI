"""
-> The phenomenon of cycle-to-cycle fluctuations in the fundamental period is referred to variously as pitch perturbation, fundamental frequency perturbation, or vocal jitter. 

-> Speech analysis can be broken down into roughly two categories -> Acoustic Analysis -> Pitch Related -> Pitch
                                                                                                        -> Jitter
                                                                                                        -> Shimmer
                                                                                                        -> Harmonic-to-Noise Ratio (HNR)
                                                                  -> Prosodic Analysis
"""

from python_speech_features import mfcc
from parselmouth.praat import call
import matplotlib.pyplot as plt
from acoustics import Signal
from colorama import Fore
import numpy as np
import soundfile
import librosa
import os

mysp=__import__("my-voice-analysis")

class tone_analysis:
    def __init__(self):
        pass

    def load_recording(self, file_path):
        if 'VietMed' in file_path:
            self.file_path = file_path
            self.data_arr, self.fs = soundfile.read(self.file_path)
            self.signal = Signal(self.data_arr, fs=self.fs)

    def load_folder(self, folder_dir):
        if 'VietMed' in folder_dir:
            self.signal = np.array([])
            for file in os.listdir(folder_dir):
                filename = os.fsdecode(file)
                self.data_arr, self.fs = soundfile.read(folder_dir + '/' + filename)
                self.signal = np.concatenate((self.signal, np.array(self.data_arr)))

            self.signal = Signal(self.signal, fs=self.fs)

    def get_tempo(self):
        self.tempo = librosa.feature.tempo(y=self.signal, sr=self.fs)[0]
        return self.tempo
    
    def get_flatness(self):     # A value closer to 1 means it is similar to white noise. A value closer to 0 is flatter.
        self.spec_flatness = np.mean(np.array(librosa.feature.spectral_flatness(y=self.signal)))
        return self.spec_flatness

    def get_mfcc(self, sample_rate=8000):
        self.mfcc_result = np.array(mfcc(self.signal, samplerate=sample_rate, ))
        return self.mfcc_result
    
    def get_f0(self, print_avg=False):
        self.f0 = np.array(librosa.yin(self.signal, fmin = librosa.note_to_hz('C2'), fmax= librosa.note_to_hz('C7'), frame_length=400))
        self.f0_mean = np.mean(self.f0)
        if print_avg:
            print(f'f0 mean: {self.f0_mean}')

        return self.f0, self.f0_mean

    def plot_f0(self, plot_avg=False, window=5):
        # plt.plot(self.mfcc_result[:, [0]])
        # plt.plot(self.mfcc_result[0])
        # print(f0.shape)
        # print(f0)
        if plot_avg:
            self.f0_avg = np.concatenate((np.full(window-1,np.nan), np.convolve(self.f0, np.ones(window), 'valid') / window))
            plt.plot(self.f0_avg, linewidth=1)

        plt.plot(self.f0, linewidth=1)
        plt.show()

    def get_analysis_results(self, f0min=65, f0max=2093):
        harmonicity = call(self.signal, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        pointProcess = call(self.signal, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer =  call([self.signal, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        localdbShimmer = call([self.signal, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3Shimmer = call([self.signal, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        aqpq5Shimmer = call([self.signal, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11Shimmer =  call([self.signal, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        ddaShimmer = call([self.signal, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        print(f'Local Shimmer: {localShimmer}')


if __name__ == '__main__':

    folder_path = 'VietMed/labeled_medical_data/cv_audio/cv_audio/VietMed_007'

    tempo_li = []
    flatness_li = []
    f0_arr = np.array([])

    for file in os.listdir(folder_path):
        analysis1 = tone_analysis()
        analysis1.load_recording(folder_path + '/' + os.fsdecode(file))
        # analysis1.load_folder('VietMed/labeled_medical_data/cv_audio/cv_audio/VietMed_007')
        analysis1.get_f0(True)
        # analysis1.plot_f0(True, 25)
        print(f'Tempo: {analysis1.get_tempo()}')
        print(f'Spectral Flatness: {analysis1.get_flatness()}')

        f0_arr = np.concatenate((f0_arr, analysis1.f0))
        tempo_li.append(analysis1.tempo)
        flatness_li.append(analysis1.spec_flatness)

    print(Fore.CYAN)
    analysis2 = tone_analysis()
    analysis2.load_folder(folder_path)
    analysis2.get_f0(True)
    # analysis1.plot_f0(True, 25)
    print(f'Tempo Mean: {analysis1.get_tempo()}')
    print(f'Mean Spectral Flatness: {analysis1.get_flatness()}')


    fig, ax = plt.subplots()
    ax.plot(f0_arr, color='red', linewidth=0.5)
    ax.tick_params(axis='y', labelcolor='red')

    # ax2 = ax.twinx()
    # ax2.plot(tempo_li, color='green')
    # ax2.tick_params(axis='y', labelcolor='green')

    plt.show()