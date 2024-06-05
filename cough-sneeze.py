from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from scipy.io import wavfile
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import librosa
import audb
import math
import os

class cough_sneeze_gmm:
    def __init__(self):
        self.data_path = 'cough-speech-sneeze/2.0.1/d3b62a9b/'

    def load_data(self):
        # If data is in the local directory then load the csv info file.
        # If the data isn't in the local directory it will be downloaded to the library folder, after which it will need manually moving.
        
        file_paths = []
        if os.path.exists('cough-speech-sneeze'):
            self.file_paths = pd.read_csv(self.data_path + 'db.files.csv')
            self.file_paths = self.file_paths.dropna(how='any')
            self.file_paths = self.file_paths.sample(frac=1)
        else:
            audb.available(only_latest=True)
            db = audb.load('cough-speech-sneeze',version='2.0.1')

    def split_audio(self, audio_arr, sample_rate=44100, segment_len=0.5):
        # Takes an array of a recording, and splits it into 0.5 second segments, returning a 2d array

        # print(f'Number of samples in total recording: {len(audio_arr)}')
        audio_len = len(audio_arr) / sample_rate
        segment_count = math.floor(audio_len / segment_len)
        segment_samples = math.floor(segment_len * sample_rate)
        segment_offset = math.floor((len(audio_arr) - (segment_count * segment_samples)) / 2)
        seg_list = []
        for i in range(segment_count):
            # print(f'Segment Samples: {segment_samples}')
            # print(f'Segment start and end points: {segment_offset + (i * segment_samples)}, {segment_offset + ((i + 1) * segment_samples)}')
            seg = np.array(audio_arr[segment_offset + (i * segment_samples) : segment_offset + ((i + 1) * segment_samples)])
            seg_list.append(seg)

        return seg_list

    def load_audio_arr(self, filepath):
        # Takes file path to a .wav file and returns an array of the recording.

        self.cur_filepath = filepath
        samplerate, data = wavfile.read(self.data_path + filepath)
        data = data.astype(float)

        return samplerate, data

    def audio_to_spectogram(self, audio_arr, samplerate=44100, plot=True):
        # Takes as input an audio array and returns a 2d array of a spectogram representing the audio

        # print(f'Sample Rate: {samplerate}')
        try:
            spectogram = librosa.feature.melspectrogram(y=audio_arr, sr=samplerate, n_mels=128, n_fft=2048, hop_length=128)
            # if spectogram.shape != (128, 173):
            #     print(f'Spectogram Shape: {spectogram.shape}')
            #     print(len(audio_arr))
            #     print(samplerate)

            
            if plot:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel spectrogram - ' + self.cur_filepath)
                plt.tight_layout()
                plt.show()

            return spectogram
        except:
            print(audio_arr)

        # print(f'Sample Rate: {samplerate}')
        # print(data[:10])

    def test_train_split(self, split=0.8):
        # Takes the csv data file that has already been shuffled, and splits it based on the the split value.

        data_row_count = self.file_paths.shape[0]
        split_val = round(data_row_count * split)

        self.file_paths = pd.get_dummies(self.file_paths, columns=['category'], dtype=float)         # One hot encoding

        x = self.file_paths.iloc[:, :2]
        y = self.file_paths.iloc[:, 2:]

        self.train_filepaths = x[:split_val]
        self.train_y = y[:split_val]
        self.test_filepaths = x[split_val:]
        self.test_y = y[split_val:]

        # Convert y dataframes to numpy arrays
        self.train_y = self.train_y[['category_coughing', 'category_silence', 'category_sneezing', 'category_speech']].to_numpy()
        self.test_y = self.test_y[['category_coughing', 'category_silence', 'category_sneezing', 'category_speech']].to_numpy()

    def load_train_data(self, save=False, load_saved_file=False):
        # Normalise the data
        if load_saved_file:
            with open('np_arrays/train_x.npy', 'rb') as f:
                self.train_x = np.load(f)
            with open('np_arrays/train_y.npy', 'rb') as f:
                self.train_y = np.load(f)
        else:
            self.train_x = []
            for row in self.train_filepaths.index:
                samplerate, data = self.load_audio_arr(self.file_paths['file'][row])
                self.train_x.append(self.audio_to_spectogram(self.split_audio(data, samplerate, 0.5)[0], samplerate, plot=False))
                if self.train_x[-1].shape != (128, 173):
                    print(self.file_paths['file'][row])
            
            self.train_x = np.array(self.train_x)

        if save and not load_saved_file:
            with open('np_arrays/train_x.npy', 'wb') as f:
                np.save(f, self.train_x)
            with open('np_arrays/train_y.npy', 'wb') as f:
                np.save(f, self.train_y)

        # arr_len = len(self.train_x)
        # print(self.train_x.shape)
        # self.train_x = np.array(self.train_x).reshape(arr_len, 128, 173, 1)

    def load_test_data(self, save=False, load_saved_file=False):
        # Normalise the data
        if load_saved_file:
            with open('np_arrays/test_x.npy', 'rb') as f:
                self.test_x = np.load(f)
            with open('np_arrays/test_y.npy', 'rb') as f:
                self.test_y = np.load(f)
        else:
            self.test_x = []
            for row in self.test_filepaths.index:
                samplerate, data = self.load_audio_arr(self.file_paths['file'][row])
                self.test_x.append(self.audio_to_spectogram(self.split_audio(data, samplerate, 0.5)[0], samplerate, plot=False))

            self.test_x = np.array(self.test_x)

        if save and not load_saved_file:
            with open('np_arrays/test_x.npy', 'wb') as f:
                np.save(f, self.test_x)
            with open('np_arrays/test_y.npy', 'wb') as f:
                np.save(f, self.test_y)

        # arr_len = len(self.test_x)
        # print(self.test_x.shape)
        # self.test_x = np.array(self.test_x).reshape(arr_len, 128, 173, 1)

    def get_data_overview(self):
        lengths_li = []
        for row in self.file_paths.index:
            samplerate, data = wavfile.read(self.data_path + self.file_paths['file'][row])
            lengths_li.append(data.size / samplerate)
        
        length_arr = np.array(lengths_li)
        print(f'Average recording length: {np.mean(length_arr)}')
        print(f'Min recording length: {np.min(length_arr)}')
        print(f'Max recording length: {np.max(length_arr)}')

    def build_model(self):
        input_shape = (128, 173, 1)
        self.CNNmodel = models.Sequential()
        self.CNNmodel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.CNNmodel.add(layers.MaxPooling2D((2, 2)))
        self.CNNmodel.add(layers.Dropout(0.2))
        self.CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.CNNmodel.add(layers.MaxPooling2D((2, 2)))
        self.CNNmodel.add(layers.Dropout(0.2))
        self.CNNmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.CNNmodel.add(layers.Flatten())
        self.CNNmodel.add(layers.Dense(64, activation='relu'))
        self.CNNmodel.add(layers.Dropout(0.2))
        self.CNNmodel.add(layers.Dense(32, activation='relu'))
        self.CNNmodel.add(layers.Dense(4, activation='softmax'))

    def compile_model(self):
        self.CNNmodel.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
        self.history = self.CNNmodel.fit(self.train_x, self.train_y, epochs=20, validation_data= (self.test_x, self.test_y))

    def eval_model(self):
        history_dict = self.history.history
        loss_values = history_dict['loss']
        acc_values = history_dict['accuracy']
        val_loss_values = history_dict['val_loss']
        val_acc_values = history_dict['val_accuracy']
        epochs = range(1,21)
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
        ax1.plot(epochs,loss_values,'bo',label='Training Loss')
        ax1.plot(epochs,val_loss_values,'orange', label='Validation Loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(epochs,acc_values,'bo', label='Training accuracy')
        ax2.plot(epochs,val_acc_values,'orange',label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.show()

    def plot_history(self):
        history_dict = self.history.history
        loss_values = history_dict['loss']
        acc_values = history_dict['accuracy']
        val_loss_values = history_dict['val_loss']
        val_acc_values = history_dict['val_accuracy']
        epochs = range(1,21)
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
        ax1.plot(epochs,loss_values,'bo',label='Training Loss')
        ax1.plot(epochs,val_loss_values,'orange', label='Validation Loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(epochs,acc_values,'bo', label='Training accuracy')
        ax2.plot(epochs,val_acc_values,'orange',label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.show()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    model = cough_sneeze_gmm()
    model.load_data()
    model.test_train_split(0.8)
    model.load_train_data(True, False)
    model.load_test_data(True, False)
    # samplerate, data = model.load_audio_arr(model.file_paths['file'].iloc[0])
    # split_data = model.split_audio(data, samplerate, 0.5)
    # model.audio_to_spectogram(split_data[0], samplerate, plot=True)
    # print(f'Number of Segments: {len(split_data)}')
    # model.get_data_overview()
    # model.build_model()
    # model.compile_model()
    # model.eval_model()