"""
This program is applied to the VOICED Database.
"""

import os
import wfdb
import numpy as np
from preprocessor import preprocessor
from data_proc import stft
import librosa
from sklearn.mixture import GaussianMixture
from numpy.random import default_rng
from python_speech_features import mfcc
from collections import defaultdict


directory = 'data'

class voice:
    def __init__(self, filename):
        self.filename = filename
        self.record_path = os.path.join(directory, filename)
        self.info_path = self.record_path+"-info.txt"
        self.preprocessor = preprocessor(self.info_path)
        self.SFT = stft()
        
        self.info = self.preprocessor.get_info()
        self.diagnosis = self.get_diagnosis()
        self.audio = self.get_audio_array()
        self.spectrogram = self.SFT.get_spectrogram(self.audio)
        self.mfccs = self.get_mfccs()
        self.comb_mfccs = self.get_mfccs(deltas=True)
    
    def get_diagnosis(self):
        d = self.info["Diagnosis"]
        if d.startswith('hypokinetic dysphonia'):
            d='hypokinetic dysphonia'
        elif d.startswith('hyperkinetic dysphonia'):
            d='hyperkinetic dysphonia'
        return d

    def get_audio_array(self):
        record = wfdb.rdrecord(self.record_path) 
        record_attributes = record.__dict__
        audio = np.squeeze(np.asarray(record_attributes["p_signal"]))
        return audio
    
    def plot_spectrogram(self):
        self.SFT.plot_spectrogram(self.audio,self.filename)

    def get_mfccs(self, deltas=False):
        sr = 8000   # sampling rate
        n = 20
        mfccs = librosa.feature.mfcc(y=self.audio, n_mfcc=n, sr=sr)
        if mfccs.shape != (n,75):
            mfccs = np.resize(mfccs.transpose(),(75,n)).transpose()
        if not deltas:
            return mfccs
        
        delta_MFCC = np.zeros(mfccs.shape)
        for t in range(delta_MFCC.shape[1]):
            index_t_minus_one,index_t_plus_one=t-1,t+1
            
            if index_t_minus_one<0:    
                index_t_minus_one=0
            if index_t_plus_one>=delta_MFCC.shape[1]:
                index_t_plus_one=delta_MFCC.shape[1]-1
            
            delta_MFCC[:,t]=0.5*(mfccs[:,index_t_plus_one]-mfccs[:,index_t_minus_one])

        double_delta_MFCC = np.zeros(mfccs.shape)
        for t in range(double_delta_MFCC.shape[1]):
            
            index_t_minus_one,index_t_plus_one, index_t_plus_two,index_t_minus_two=t-1,t+1,t+2,t-2
            
            if index_t_minus_one<0:
                index_t_minus_one=0
            if index_t_plus_one>=delta_MFCC.shape[1]:
                index_t_plus_one=delta_MFCC.shape[1]-1
            if index_t_minus_two<0:
                index_t_minus_two=0
            if index_t_plus_two>=delta_MFCC.shape[1]:
                index_t_plus_two=delta_MFCC.shape[1]-1
            
            double_delta_MFCC[:,t]=0.1*( 2*mfccs[:,index_t_plus_two]+mfccs[:,index_t_plus_one]
                                        -mfccs[:,index_t_minus_one]-2*mfccs[:,index_t_minus_two] )
        
        Combined_MFCC_F = np.concatenate((mfccs,delta_MFCC,double_delta_MFCC),axis=1)
        
        return Combined_MFCC_F.transpose()


class gmm_classifier:
    def __init__(self, trainset):
        # get all features
        allfeatures = defaultdict(list)
        for filename in trainset:
            voice_sample = voice(filename)
            x = voice_sample.comb_mfccs
            y = voice_sample.diagnosis
            allfeatures[y].extend(x)
        
        # normalisation stats
        allconcat = np.vstack(list(allfeatures.values()))
        self.means = np.mean(allconcat, 0)
        self.invstds = np.std(allconcat, 0)
        for i,val in enumerate(self.invstds):
            if val == 0.0:
                self.invstds[i] = 1.0
            else:
                self.invstds[i] = 1.0 / val

        # for each label, compile a normalised concatenated list of features
        aggfeatures = {}
        for label, features in allfeatures.items():
            normed = self._normalise(features)
            if label not in aggfeatures:
                aggfeatures[label] = normed
            else:
                aggfeatures[label] = np.vstack((aggfeatures[label], normed))

        # for each label's aggregated features, train a set of GMMs
        self.gmms = {}
        for label, aggf in aggfeatures.items():
            self.gmms[label] = GaussianMixture(n_components=10, init_params='k-means++', random_state=0)
            self.gmms[label].fit(aggf)
        # print(self.gmms)

    def _normalise(self, data):
        return (data - self.means) * self.invstds
    
    def classify(self, filename):
        voice_sample = voice(filename)
        mfcc_feat = voice_sample.comb_mfccs
		# For each label GMM, find the overall log-likelihood and choose the strongest
        bestlabel = ''
        bestll = -9e99
        for label, gmm in self.gmms.items():
            ll = gmm.score_samples(mfcc_feat)
            ll = np.sum(ll)
            if ll > bestll:
                bestll = ll
                bestlabel = label
        return bestlabel


# -------------------------- Split Datasets -----------------------------

def k_fold_split(n_splits, n_instances, random_generator=default_rng(seed=42)):
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)
    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices

# split into train and test sets (or used for splitting trainval set into train and val for pruning)
def split_train_test_dataset(y, n_folds, random_generator=default_rng(seed=42)):
    y=np.array(y)
    n_instances = len(y)
    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)
    
    folds = []
    for k in range(n_folds):
        # obtain indices for each part
        test_indices = split_indices[k]
        train_indices = np.setdiff1d(np.arange(n_instances), test_indices)

        # seperate dataset according to the indeces
        y_train = y[train_indices]
        y_test = y[test_indices]

        # put into list
        folds.append([y_train, y_test])
    return folds


# ------------------ evaluation ---------------
def confusion_matrix(y_gold, y_prediction, class_labels=None):
    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
        print(class_labels)
        
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (row, label_row) in enumerate(class_labels):
        class_actual = (y_gold == label_row)
        for (col, label_col) in enumerate(class_labels):
            class_predict = (y_prediction == label_col)
            check = np.logical_and(class_actual, class_predict)
            sum = np.sum(check)
            confusion[row][col] = sum

    # normalising confusion matrix
    class_actual_sum = np.sum(confusion, axis=1)
    class_actual_sum = np.transpose(np.expand_dims(class_actual_sum, axis=0))
    norm_confusion = np.divide(confusion, class_actual_sum)
    return (confusion, norm_confusion)

def accuracy_from_confusion(confusion):
    if np.sum(confusion) > 0:
        return np.sum(np.diagonal(confusion))/np.sum(confusion)
    else:
        return 0.

def precision(confusion):
    # Compute the precision per class
    try:
        p = np.diag(confusion)/np.sum(confusion, axis=0)
    except:
        p = 0

    # Compute the macro-averaged precision
    macro_p=0
    if len(p)>0:
        macro_p = np.mean(p)

    return (p, macro_p)

def recall(confusion):
    # Compute the recall per class
    try:
        r = np.diag(confusion)/np.sum(confusion, axis=1)
    except:
        r = 0

    # Compute the macro-averaged recall
    macro_r=0
    if len(r)>0:
        macro_r = np.mean(r)

    return (r, macro_r)

def f1_score(precisions, recalls):
    # to make sure they are of the same length
    assert len(precisions) == len(recalls)

    # Complete this to compute the per-class F1
    f = 2*precisions*recalls/(precisions+recalls)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)

    return (f, macro_f)

def evaluate(y_gold, y_pred):
    # check results with y_test
    confusion, norm_confusion = confusion_matrix(y_gold, y_pred)

    accuracy = accuracy_from_confusion(confusion)
    p, macro_p = precision(confusion)
    r, macro_r = recall(confusion)
    f, macro_f = f1_score(p, r)
    return(confusion, norm_confusion, accuracy, p, r, f)


if __name__ == "__main__":
    # filename = "voice001"
    # voice_sample = voice(filename)
    # mfcc_feat = mfcc(voice_sample.audio, 8000)
    # print(np.shape(mfcc_feat))

    file_names = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            if f[-8:] == 'info.txt':
                file_names.append(filename[:8])
    # print(file_names)

    count_dict = {}
    trainset = []
    testset = []
    for filename in file_names:
        voice_sample = voice(filename)
        y = voice_sample.diagnosis
        if y not in count_dict:
            count_dict[y] = 1
        else:
            count_dict[y] += 1
        if count_dict[y] <= 30:
            trainset.append(filename)
        else:
            testset.append(filename)

    # trainset, testset = split_train_test_dataset(file_names, 10)[0]


    gmm = gmm_classifier(trainset)

    # testing
    y_gold = []
    y_pred = []
    for filename in testset:
        gold_label = voice(filename).diagnosis
        pred_label = gmm.classify(filename)
        y_gold.append(gold_label)
        y_pred.append(pred_label)
    y_gold = np.array(y_gold)
    y_pred = np.array(y_pred)

    confusion, norm_confusion = confusion_matrix(y_gold, y_pred)
    print(confusion)

