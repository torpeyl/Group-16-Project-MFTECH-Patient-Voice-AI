"""
This program is applied to the Saarbruecken Voice Database (SVD).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from collections import defaultdict
from sklearn.mixture import GaussianMixture

directory = 'data_svd'
# subdirectory = 'healthy'
filename_suffix = '-a_n.wav'


class voice_svd:
    def __init__(self, filename, label):
        self.filename = filename
        self.label = label
        self.record_path = os.path.join(directory, label, filename)
        
        self.audio, self.sample_rate = librosa.load(self.record_path)

        self.mfccs = self.get_mfccs()
        self.comb_mfccs = self.get_mfccs(deltas=True)

    def get_mfccs(self, deltas=False):
        """
            Code adapted from https://github.com/genzen2103/Speaker-Recognition-System-using-GMM/blob/master/MFCC_speaker_recog.py
        """
        sr = self.sample_rate
        n = 20

        # calculate MFCCs
        mfccs = librosa.feature.mfcc(y=self.audio, n_mfcc=n, sr=sr)
        if not deltas:
            return mfccs.transpose()        # shape: (len,n)
        
        # calculate delta MFCCs
        delta_MFCC = np.zeros(mfccs.shape)
        for t in range(delta_MFCC.shape[1]):
            index_t_minus_one,index_t_plus_one=t-1,t+1
            
            if index_t_minus_one<0:    
                index_t_minus_one=0
            if index_t_plus_one>=delta_MFCC.shape[1]:
                index_t_plus_one=delta_MFCC.shape[1]-1
            
            delta_MFCC[:,t]=0.5*(mfccs[:,index_t_plus_one]-mfccs[:,index_t_minus_one])

        # calculate double delta MFCCs
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
        
        return Combined_MFCC_F.transpose()       # shape: (len,n)


class gmm_classifier:
    """
        Code adapted from https://github.com/danstowell/smacpy/blob/main/smacpy.py
    """
    def __init__(self, trainset,load_gmms=False,load_features=False,n_components=4):
        f = 'features.npy'
        self.gmms = {}
        if load_gmms:   # use existing GMMs directly
            # load dataset stats
            print("loading existing dataset stats...")
            self.means = np.load(os.path.join('gmms','data_avgs.npy'))
            self.invstds = np.load(os.path.join('gmms','data_invstds.npy'))
            
            # load gmms
            print("loading existing GMMs...")
            gmm_names = []
            for filename in os.listdir("gmms"):
                file_path = os.path.join("gmms", filename)
                if os.path.isfile(file_path):
                    if file_path[-10:] == '_means.npy':
                        gmm_names.append(filename[:-10])
            # print(gmm_names)  # debug
            for gmm_name in gmm_names:
                means = np.load(os.path.join('gmms', gmm_name + '_means.npy'))
                covar = np.load(os.path.join('gmms', gmm_name + '_covariances.npy'))
                gmm = GaussianMixture(n_components = len(means), covariance_type='full')
                gmm.precisions_cholesky_ = np.load(os.path.join('gmms', gmm_name + '_precisions_cholesky.npy'))
                gmm.weights_ = np.load(os.path.join('gmms', gmm_name + '_weights.npy'))
                gmm.means_ = means
                gmm.covariances_ = covar
                self.gmms[gmm_name] = gmm
            print(self.gmms)  # debug

        else:       # fit new GMMs
            if load_features and os.path.isfile(f):   # no change to MFCCs
                print("loading existing MFCC features...")
                aggfeatures = np.load(f,allow_pickle='TRUE').item()

                # load dataset stats
                print("loading existing dataset stats...")
                self.means = np.load(os.path.join('gmms','data_avgs.npy'))
                self.invstds = np.load(os.path.join('gmms','data_invstds.npy'))
            else:                                     # generate new MFCCs
                # get all MFCC features
                print("calculating MFCC features...")
                allfeatures = defaultdict(list)
                for filename,label in trainset:
                    voice_sample = voice_svd(filename,label)
                    x = voice_sample.comb_mfccs
                    y = voice_sample.label
                    allfeatures[y].extend(x)
                
                # normalisation stats
                print("calculating dataset stats...")
                allconcat = np.vstack(list(allfeatures.values()))
                self.means = np.mean(allconcat, 0)
                self.invstds = np.std(allconcat, 0)
                for i,val in enumerate(self.invstds):
                    if val == 0.0:
                        self.invstds[i] = 1.0
                    else:
                        self.invstds[i] = 1.0 / val
                # save the means and invstads
                np.save(os.path.join('gmms','data_avgs'), self.means)
                np.save(os.path.join('gmms','data_invstds'), self.invstds)

                # for each label, compile a normalised concatenated list of features
                aggfeatures = {}
                for label, features in allfeatures.items():
                    normed = self._normalise(features)
                    if label not in aggfeatures:
                        aggfeatures[label] = normed
                    else:
                        aggfeatures[label] = np.vstack((aggfeatures[label], normed))
                np.save(f, aggfeatures) 

            # for each label's aggregated features, train a set of GMMs
            print("fitting GMMs...")
            for label, aggf in aggfeatures.items():
                gmm = GaussianMixture(n_components=n_components, init_params='k-means++', random_state=0)
                gmm.fit(aggf)
                self.gmms[label] = gmm
                np.save(os.path.join('gmms', label + '_weights'), gmm.weights_, allow_pickle=False)
                np.save(os.path.join('gmms', label + '_means'), gmm.means_, allow_pickle=False)
                np.save(os.path.join('gmms', label + '_covariances'), gmm.covariances_, allow_pickle=False)
                np.save(os.path.join('gmms', label + '_precisions_cholesky'), gmm.precisions_cholesky_, allow_pickle=False)
            # print(self.gmms)
    
    def _normalise(self, data):
        return (data - self.means) * self.invstds
    
    def classify(self, filename, label):
        voice_sample = voice_svd(filename, label)
        mfcc_feat = self._normalise(voice_sample.comb_mfccs)
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


def demo(filename="2-a_n.wav", label="healthy"):
    voice_sample = voice_svd(filename,label)
    # print(voice_sample.audio)
    mfccs = voice_sample.mfccs.transpose()
    sr = voice_sample.sample_rate
    # print(np.shape(mfccs))
    # print(sr)

    # plot the MFCCs and save the image
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mfccs, 
                            x_axis="time", 
                            sr=sr)
    plt.colorbar(format="%+2.f")
    plt.savefig(filename+".png")


# ------------------ dataset initialisation ---------------
def get_dataset(label):
    file_names = []
    folder = os.path.join(directory, label)

    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        if os.path.isfile(f):
            file_names.append(filename)
    return file_names

def split_dataset(dataset, train_percentage):
    trainset = []
    testset = []
    # split the sets
    for label, data in dataset.items():
        train_index = np.ceil(len(data)*train_percentage).astype('int')
        trainset.extend(zip(data[:train_index], np.repeat(label,len(data[:train_index]))))
        testset.extend(zip(data[train_index:], np.repeat(label,len(data[train_index:]))))
    # shuffle the sets
    np.random.shuffle(trainset)
    np.random.shuffle(testset)
    return trainset, testset


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
    np.random.seed(42) 
    dataset_size = 600
    train_percentage = 0.8

    # initialise the dataset
    dataset = {}
    classes = ["healthy", "pathological"]
    for label in classes:
        data = get_dataset(label)
        np.random.shuffle(data)          # shuffle the order
        dataset[label] = data[:dataset_size]    # only take the fixed size

    # split into train-test sets
    trainset, testset = split_dataset(dataset, train_percentage)

    # train GMMs
    gmm = gmm_classifier(trainset,load_gmms=False,load_features=True,n_components=8)

    # testing
    y_gold = []
    y_pred = []
    for filename,label in testset:
        gold_label = label
        pred_label = gmm.classify(filename, label)
        y_gold.append(gold_label)
        y_pred.append(pred_label)
    y_gold = np.array(y_gold)
    y_pred = np.array(y_pred)

    # evaluate the results
    confusion, norm_confusion = confusion_matrix(y_gold, y_pred)
    print("confusion matrix:\n",confusion)
    accuracy = accuracy_from_confusion(confusion)
    print("accuracy: ",accuracy)

    