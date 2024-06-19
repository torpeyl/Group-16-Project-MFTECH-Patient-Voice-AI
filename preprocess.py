import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft

class Preprocess:
    def __init__(self, directory, pre_emphasis_coefficient=0.97, apply_compression=True):
        self.directory = directory
        self.pre_emphasis_coefficient = pre_emphasis_coefficient
        self.apply_compression = apply_compression
        self.emotions = ["a", "d", "f", "h", "sa", "su", "n"]
        self.emotion_mapping = {
            "a": "anger",
            "d": "disgust",
            "f": "fear",
            "h": "happiness",
            "sa": "sadness",
            "su": "surprise",
            "n": "neutral"
        }
        self.emotion_to_int = {emotion: idx for idx, emotion in enumerate(self.emotion_mapping.values())}
        self.first_window_printed = False

    def pre_emphasis_librosa(self, signal):
        return librosa.effects.preemphasis(signal, coef=self.pre_emphasis_coefficient)

    def cubic_root_compression(self, signal):
        return np.sign(signal) * np.abs(signal)**(2/3)

    def augment_data(self, signal, sr):
        augmented_signals = []
        # Time Stretching
        stretched = librosa.effects.time_stretch(signal, rate=1.1)
        augmented_signals.append(stretched)
        # Pitch Shifting
        pitch_shifted = librosa.effects.pitch_shift(signal, sr=sr, n_steps=2)
        augmented_signals.append(pitch_shifted)
        # Noise Injection
        noise = np.random.randn(len(signal))
        signal_noise = signal + 0.005 * noise
        augmented_signals.append(signal_noise)
        return augmented_signals

    def compute_features(self, signal, sr):
        features = []
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        frames = librosa.util.frame(signal, frame_length=len(mfccs[0]), hop_length=len(mfccs[0])//2)

        for i, frame in enumerate(frames):
            spectrum = np.abs(fft(frame))[:len(frame)//2]
            rmse = np.sqrt(np.mean(frame**2))
            zcr = ((frame[:-1] * frame[1:]) < 0).sum() / len(frame)
            mfcc = mfccs[:, i]
            features.append((spectrum, rmse, zcr, mfcc))
            if i == 0 and not self.first_window_printed:
                print("RMSE of the first window:", rmse)
                print("ZCR of the first window:", zcr)
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(spectrum)
                plt.title('DFT Spectrum')
                plt.xlabel('Frequency')
                plt.ylabel('Amplitude')

                plt.subplot(1, 2, 2)
                librosa.display.specshow(mfccs, sr=sr, x_axis='time')
                plt.colorbar()
                plt.title('MFCC')
                plt.tight_layout()
                plt.show()
                self.first_window_printed = True
        return features

    def compute_graph_features(self, features):
        # Include rmse and zcr in node features
        node_features = [np.concatenate((f[3], [f[1], f[2]])) for f in features]
        num_nodes = len(node_features)
        adjacency_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
        return np.array(node_features), adjacency_matrix

    def load_data(self):
        data = []
        original_data = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".wav"):
                label = filename.split('_')[1]
                label = ''.join(filter(str.isalpha, label.split('.')[0]))
                audio_data, sr = librosa.load(os.path.join(self.directory, filename), sr=None)
                original_data.append((audio_data, sr))

                # Augment data
                augmented_signals = self.augment_data(audio_data, sr)
                augmented_signals.append(audio_data)  # Include original signal

                for signal in augmented_signals:
                    pre_emphasized_audio_librosa = self.pre_emphasis_librosa(signal)
                    if self.apply_compression:
                        pre_emphasized_audio_librosa = self.cubic_root_compression(pre_emphasized_audio_librosa)

                    features_librosa = self.compute_features(pre_emphasized_audio_librosa, sr)
                    node_features, adjacency_matrix = self.compute_graph_features(features_librosa)

                    if label in self.emotion_mapping:
                        data.append({
                            "filename": filename,
                            "emotion": self.emotion_mapping[label],
                            "node_features": node_features,
                            "adjacency_matrix": adjacency_matrix,
                            "sr": sr
                        })

        self.df = pd.DataFrame(data)
        self.original_data = original_data
        print(len(data))

    def one_hot_encode_emotions(self):
        if 'emotion' in self.df.columns:
            self.df['emotion_int'] = self.df['emotion'].apply(lambda x: self.emotion_to_int[x])
            self.one_hot_encoded = pd.get_dummies(self.df['emotion'], columns=['emotion'])
            self.df = pd.concat([self.df, self.one_hot_encoded], axis=1)
        else:
            raise ValueError("No emotion column")

    def get_processed_data(self):
        return self.df
