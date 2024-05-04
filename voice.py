import os
import wfdb
import numpy as np
from preprocessor import preprocessor
from data_proc import stft

directory = 'data'

class voice:
    def __init__(self, filename):
        self.filename = filename
        self.record_path = os.path.join(directory, filename)
        self.info_path = self.record_path+"-info.txt"
        self.preprocessor = preprocessor(self.info_path)
        self.SFT = stft()
        
        self.info = self.preprocessor.get_info()
        self.audio = self.get_audio_array()
        self.spectrogram = self.SFT.get_spectrogram(self.audio)
    
    def get_audio_array(self):
        record = wfdb.rdrecord(self.record_path) 
        record_attributes = record.__dict__
        audio = np.squeeze(np.asarray(record_attributes["p_signal"]))
        return audio
    
    def plot_spectrogram(self):
        self.SFT.plot_spectrogram(self.audio,self.filename)


if __name__ == "__main__":
    filename = "voice001"
    voice_sample = voice(filename)
    print(voice_sample.info)
    voice_sample.plot_spectrogram()