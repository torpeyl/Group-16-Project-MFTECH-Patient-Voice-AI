
# !pip install tensorflow_io
import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from keras import layers as L
from keras import backend as K
from keras import Model
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

model_dir = "./Cough_Model/saved_models"

## define parameters
classes = ["coughing", "sneezing", "speech", "silence"]
label_index = {
    "coughing":0,
    "sneezing":1,
    "speech":2,
    "silence":3
}
sample_rate = 22050   # default sample rate in librosa
chunk_duration = 1.5  # in seconds
chunksize = int(chunk_duration*sample_rate)
frame_duration = 0.75  # in seconds


def preprocess(x, chunksize):
    x_len = len(x)
    if x_len < chunksize:   # padding
        # processed = np.pad(x, (0,chunksize-x_len))  # right-pad with zero
        processed = np.resize(x, (chunksize,))        # repeat padding
    elif x_len > chunksize: # splitting
        tailing = x_len % chunksize
        if tailing != 0:                # handle uneven splits: create overlap for the last chunk
            x = np.concatenate((x[:-tailing], x[-chunksize:]))
        num_chunks = int(round(len(x)/chunksize))
        processed = np.array_split(x,num_chunks)
    else:                   # no change
        processed = np.array(x)
    return processed


def get_normalised_mel_spectrogram(x, sr=sample_rate, n_mel_bins=20):
    """
          Code adapted from https://github.com/douglas125/SpeechCmdRecognition/blob/master/audioUtils.py#L93
    """
    spec_stride = 128
    spec_len = 1024

    spectrogram = tfio.audio.spectrogram(
        x, nfft=spec_len, window=spec_len, stride=spec_stride
    )

    num_spectrogram_bins = spec_len // 2 + 1  # spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz = 40.0, 8000.0
    num_mel_bins = n_mel_bins
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    avg = tf.math.reduce_mean(log_mel_spectrograms)
    std = tf.math.reduce_std(log_mel_spectrograms)

    return (log_mel_spectrograms - avg) / std


def get_melspec_model(iLen=None):
    inp = L.Input((iLen,), name='input')
    mel_spec = get_normalised_mel_spectrogram(inp)
    melspecModel = Model(inputs=inp, outputs=mel_spec, name='normalised_spectrogram_model')
    return melspecModel


class RNNSpeechModel:
    def __init__(self, classes=classes, sampling_rate=sample_rate, input_length=chunksize, rnn_func=L.LSTM):
        self.classes = np.asarray(classes)
        self.nCategs = len(self.classes)
        self.sr = sampling_rate
        self.iLen = input_length
        self.rnn_func = rnn_func
        self.model_name = 'model-spec-RNN-90s.h5'
        self.model = self._create_model()


    def _create_model(self):
        inputs = L.Input((self.iLen,), name='input')

        m =  get_melspec_model(iLen=self.iLen)
        m.trainable = False

        x = m(inputs)
        x = tf.expand_dims(x, axis=-1, name='mel_stft')
        x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
        x = L.BatchNormalization()(x)
        x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
        x = L.BatchNormalization()(x)

        x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

        x = L.Bidirectional(self.rnn_func(64, return_sequences=True))(x)
        x = L.Bidirectional(self.rnn_func(64, return_sequences=True))(x)

        # for classification
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dense(64, activation='relu')(x)
        x = L.Dense(32)(x)

        output = L.Dense(self.nCategs, activation='softmax', name='output')(x)

        model = Model(inputs=[inputs], outputs=[output])
        model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
        model.load_weights(os.path.join(model_dir, self.model_name))

        return model


    def predict_framewise(self, x, frame_duration=frame_duration, chunksize=chunksize):
        # preprocess the input audio
        # 1. split into frames
        x = np.asarray(preprocess(x, int(frame_duration*sample_rate))).astype('float32')
        # 2. extract features for each frame
        new_x = []
        for x_slice in x:
            new_x_slice = np.squeeze(preprocess(x_slice, chunksize))
            new_x.append(new_x_slice)
        x = np.asarray(new_x)
        # print(np.shape(x))        # for debugging

        # feed it into the model to produce predictions
        y_pred = self.model.predict(x)
        y_pred_probs = np.max(y_pred, axis=1)   # the probability of the highest scoring class
        y_pred_categorical = np.argmax(y_pred, axis=1)
        y_pred_labels = self.classes[y_pred_categorical]   # the label of the highest scoring class
        y_pred_speech = y_pred[:,list(self.classes).index("speech")]   # probability that the segment is speech

        return y_pred_labels, y_pred_probs, y_pred_speech


"""Calculate health indicators:
1. speech ratio
2. disruption ratio
"""

## calculate label ratios as health indicators
def calculate_health_indicators(pred_labels):
    classes, counts = np.unique(pred_labels,return_counts=True)
    counts_dict = dict(zip(classes,counts))
    speech_count = counts_dict['speech'] if 'speech' in counts_dict else 0
    speech_ratio = speech_count/np.sum(counts)
    disruption_ratio = (np.sum(counts) - speech_count) / np.sum(counts)
    return speech_ratio,disruption_ratio
