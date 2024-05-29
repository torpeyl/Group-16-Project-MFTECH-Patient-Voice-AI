import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from data_proc import stft
from plotter import TimescalePlotter
import opensmile
import math


directory = "data\\cough-speech-sneeze"
classes = ["coughing", "sneezing", "speech", "silence"]


smile_functionals = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
smile_llds = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

class voice_css:
    def __init__(self, filename, label):
        self.filename = filename
        self.label = label
        self.record_path = os.path.join(directory, label, filename)
        
        self.audio, self.sample_rate = librosa.load(self.record_path)

        # self.SFT = stft(sr=self.sample_rate)
        # self.spectrogram = self.SFT.get_spectrogram(self.audio).transpose()

        # self.mfccs = self.get_mfccs()
        # self.comb_mfccs = self.get_mfccs(deltas=True)

        # define the feature vector for classification use
        self.N = 5      # stride size
        self.frame_size = 0.02*self.N   # in seconds
        self.window_duration = 1.0      # in seconds
        self.window_size = int(self.window_duration*self.sample_rate)
        self.audio_duration = len(self.audio)/self.sample_rate
        self.window_num = int(self.audio_duration/self.window_duration)
        self.feature = self._get_features_functional()
        # self.feature = self._get_features_lld()

    
    def _filter_feature(self, x):     # moving average with stride=N to reduce dimensionality
        if len(x)%2 != 0:   # odd
            x = x[:(len(x)-1)]
        new_x = np.reshape(x,(int(len(x)/2),2))
        z1 = np.add.reduceat(new_x[:,0], np.arange(0, len(new_x[:,0]), self.N))/self.N
        z2 = np.add.reduceat(new_x[:,1], np.arange(0, len(new_x[:,1]), self.N))/self.N
        z = np.vstack((z1,z2))
        return z

    # Feature Level: Low Level Descriptors
    def _get_features_lld(self):
        llds = smile_llds.process_signal(self.audio,self.sample_rate)

        ## spectral features
        mfcc1 = self._filter_feature(np.array(llds['mfcc1_sma3']))
        flux = self._filter_feature(np.array(llds['spectralFlux_sma3']))
        # h1_a3 = self._filter_feature(np.array(llds['logRelF0-H1-A3_sma3nz']))
        hammarberg =  self._filter_feature(np.array(llds['hammarbergIndex_sma3']))

        ## prosodic features
        loudness = self._filter_feature(np.array(llds['Loudness_sma3']))
        pitch = self._filter_feature(np.array(llds['F0semitoneFrom27.5Hz_sma3nz']))     # F0 envelope
        shimmer = self._filter_feature(np.array(llds['shimmerLocaldB_sma3nz']))
        jitter = self._filter_feature(np.array(llds['jitterLocal_sma3nz']))
        hnr =  self._filter_feature(np.array(llds['HNRdBACF_sma3nz']))
        f1freq = self._filter_feature(np.array(llds['F1frequency_sma3nz']))
        # f1band = self._filter_feature(np.array(llds['F1bandwidth_sma3nz']))
        # f1amp = self._filter_feature(np.array(llds['F1amplitudeLogRelF0_sma3nz']))
        f2freq = self._filter_feature(np.array(llds['F2frequency_sma3nz']))
        # f3freq = self._filter_feature(np.array(llds['F3frequency_sma3nz']))

        ## create the feature vector
        feature = np.vstack((
            mfcc1,
            flux,          # 
            # h1_a3,
            # hammarberg,
            loudness,
            pitch,          #
            jitter,
            shimmer,       # 
            # hnr,
            # f1freq,
            # f1band,
            # f1amp,
            # f2freq,
            # f3freq,
            ))
        return feature.transpose()

    # Feature Level: Functionals
    def _get_features_functional(self):
        functionals = smile_functionals.process_signal(self.audio,self.sample_rate)

        ## spectral features
        flux_mean = float(functionals['spectralFlux_sma3_amean'].iloc[0])
        flux_std = float(functionals['spectralFlux_sma3_stddevNorm'].iloc[0])
        ## prosodic features
        loudness_mean = float(functionals['loudness_sma3_amean'].iloc[0])
        loudness_std = float(functionals['loudness_sma3_stddevNorm'].iloc[0])
        pitch_mean = float(functionals['F0semitoneFrom27.5Hz_sma3nz_amean'].iloc[0])
        pitch_std = float(functionals['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'].iloc[0])
        shimmer_mean = float(functionals['shimmerLocaldB_sma3nz_amean'].iloc[0])
        shimmer_std = float(functionals['shimmerLocaldB_sma3nz_stddevNorm'].iloc[0])
        jitter_mean = float(functionals['jitterLocal_sma3nz_amean'].iloc[0])
        jitter_std = float(functionals['jitterLocal_sma3nz_stddevNorm'].iloc[0])
        hnr_mean = float(functionals['HNRdBACF_sma3nz_amean'].iloc[0])
        hnr_std = float(functionals['HNRdBACF_sma3nz_stddevNorm'].iloc[0])
        f1freq_mean = float(functionals['F1frequency_sma3nz_amean'].iloc[0])
        f1freq_std = float(functionals['F1frequency_sma3nz_stddevNorm'].iloc[0])
        mfcc1_mean = float(functionals['mfcc1_sma3_amean'].iloc[0])
        mfcc1_std = float(functionals['mfcc1_sma3_stddevNorm'].iloc[0])
        # mfcc2_mean = float(functionals['mfcc2_sma3_amean'].iloc[0])
        # mfcc2_std = float(functionals['mfcc2_sma3_stddevNorm'].iloc[0])
        # mfcc3_mean = float(functionals['mfcc3_sma3_amean'].iloc[0])
        # mfcc3_std = float(functionals['mfcc3_sma3_stddevNorm'].iloc[0])
        # mfcc4_mean = float(functionals['mfcc4_sma3_amean'].iloc[0])
        # mfcc4_std = float(functionals['mfcc4_sma3_stddevNorm'].iloc[0])
        # f1band_mean = float(functionals['F1bandwidth_sma3nz_amean'].iloc[0])
        # f1band_std = float(functionals['F1bandwidth_sma3nz_stddevNorm'].iloc[0])
        # f3_mean = float(functionals['F3frequency_sma3nz_amean'].iloc[0])
        # f3_std = float(functionals['F3frequency_sma3nz_stddevNorm'].iloc[0])

        ## create the feature vector
        feature = np.expand_dims([
            flux_mean,         #
            flux_std,          # 
            loudness_mean,     #
            loudness_std,      #
            # pitch_mean,
            # pitch_std,
            jitter_mean,       #
            jitter_std,        #
            # shimmer_mean,
            shimmer_std,       # 
            # hnr_mean,
            # hnr_std,
            # f1freq_mean,
            # f1freq_std,
            mfcc1_mean,        #
            mfcc1_std,         #
            # mfcc2_mean,
            # mfcc2_std,
            # mfcc3_mean,
            # mfcc3_std,
            # mfcc4_mean,
            # mfcc4_std,
            # f1band_mean,
            # f1band_std,
            # f3_mean,
            # f3_std,
            ], axis=0)
        return feature

    # used for extracting features during predictions
    def get_features_for_predict(self):
        sections = np.array_split(self.audio, self.window_num)
        feature_stack = []
        for section in sections:
            functionals = smile_functionals.process_signal(section,self.sample_rate)

            ## spectral features
            flux_mean = float(functionals['spectralFlux_sma3_amean'].iloc[0])
            flux_std = float(functionals['spectralFlux_sma3_stddevNorm'].iloc[0])
            mfcc1_mean = float(functionals['mfcc1_sma3_amean'].iloc[0])
            mfcc1_std = float(functionals['mfcc1_sma3_stddevNorm'].iloc[0])

            ## prosodic features
            loudness_mean = float(functionals['loudness_sma3_amean'].iloc[0])
            loudness_std = float(functionals['loudness_sma3_stddevNorm'].iloc[0])
            # pitch_mean = float(functionals['F0semitoneFrom27.5Hz_sma3nz_amean'].iloc[0])
            # pitch_std = float(functionals['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'].iloc[0])
            # shimmer_mean = float(functionals['shimmerLocaldB_sma3nz_amean'].iloc[0])
            shimmer_std = float(functionals['shimmerLocaldB_sma3nz_stddevNorm'].iloc[0])
            jitter_mean = float(functionals['jitterLocal_sma3nz_amean'].iloc[0])
            jitter_std = float(functionals['jitterLocal_sma3nz_stddevNorm'].iloc[0])
            
            ## create the feature vector
            feature = [
                flux_mean,         #
                flux_std,          # 
                loudness_mean,     #
                loudness_std,      #
                # pitch_mean,
                # pitch_std,
                jitter_mean,       #
                jitter_std,        #
                # shimmer_mean,
                shimmer_std,       # 
                mfcc1_mean,        #
                mfcc1_std,         #
            ]
            feature_stack.append(feature)
        return (np.array(feature_stack))

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
                print("loading existing audio features...")
                aggfeatures = np.load(f,allow_pickle='TRUE').item()

                # load dataset stats
                print("loading existing dataset stats...")
                self.means = np.load(os.path.join('gmms','data_avgs.npy'))
                self.invstds = np.load(os.path.join('gmms','data_invstds.npy'))
            else:                                     # generate new features
                # get all audio features
                print("calculating audio features...")
                allfeatures = defaultdict(list)
                for filename,label in trainset:
                    voice_sample = voice_css(filename,label)
                    x = voice_sample.feature
                    # x = voice_sample.spectrogram      # takes very long
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
        voice_sample = voice_css(filename, label)
        feat = self._normalise(voice_sample.feature)
        # feat = self._normalise(voice_sample.spectrogram)    # takes very long

		# For each label GMM, find the overall log-likelihood and choose the strongest
        bestlabel = ''
        bestll = -9e99
        for label, gmm in self.gmms.items():
            ll = gmm.score_samples(feat)
            ll = np.sum(ll)
            if ll > bestll:
                bestll = ll
                bestlabel = label
        return bestlabel
    
    def classify_framewise(self, filename, label):
        voice_sample = voice_css(filename, label)
        feat = self._normalise(voice_sample.get_features_for_predict())
        # feat = self._normalise(voice_sample.spectrogram)    # takes very long

		## For each label GMM, find the overall log-likelihood and choose the strongest
        predictions = []
        for feat_i in feat:
            bestlabel = ''
            bestll = -9e99
            feat_i = np.expand_dims(feat_i,axis=0)
            for label, gmm in self.gmms.items():
                ll = gmm.score_samples(feat_i)
                ll = np.sum(ll)
                if ll > bestll:
                    bestll = ll
                    bestlabel = label
            predictions.append(bestlabel)
        # generate timestamps
        timestamps = np.linspace(start=0, stop=voice_sample.audio_duration, num=len(predictions))
        return (timestamps,predictions)


def demo(filename="_0rh6xgxhrq_9.18-15.85.wav", label="coughing"):
    voice_sample = voice_css(filename,label)
    return voice_sample.feature
    # print(voice_sample.audio)
    # mfccs = voice_sample.mfccs.transpose()
    # sr = voice_sample.sample_rate
    # print(np.shape(mfccs))
    # print(sr)

    # # plot the MFCCs and save the image
    # plt.figure(figsize=(25, 10))
    # librosa.display.specshow(mfccs, 
    #                         x_axis="time", 
    #                         sr=sr)
    # plt.colorbar(format="%+2.f")
    # plt.savefig(filename+".png")


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


# ------------------ visualisation ---------------
# (unused)
def data_visualisation(dataset_size = 50):
    visualise_set = defaultdict(list)
    for label in classes:
        data = get_dataset(label)
        np.random.shuffle(data)          # shuffle the order
        filenames = data[:dataset_size]    # only take the fixed size

        for filename in filenames:
            voice_sample = voice_css(filename,label)
            x = voice_sample.feature
            visualise_set[label].extend(x)

        # visualise_set[label] = np.array(visualise_set[label])
    # print(visualise_set)

    # Create an array of indices for the x-axis
    indices = np.arange(0, np.shape(visualise_set['coughing'])[1])
    
    colour_code = {
        "coughing":'red',
        "sneezing":'blue',
        "speech":'green'
    }
    index_code = {
        "coughing":1,
        "sneezing":2,
        "speech":3
    }

    for i in indices:
        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot the data points
        for label in classes:
            for feature_values in visualise_set[label]:
                plt.scatter([index_code[label]], [feature_values[i]], color=colour_code[label])
                # plt.scatter(indices, feature_values, color=colour_code[label], label=label)

        # Add labels and title
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        plt.title('Data visualisation')
        # plt.legend()

        # Display the plot
        plt.savefig(f'{i}.png')


if __name__ == "__main__":
    # label = classes[2]
    # filename = "_kwn1b3lq90_10.7-12.01.wav" #"_kwn1b3lq90_43.6-44.11.wav" #"_0rh6xgxhrq_9.18-15.85.wav"
    # aud_path = os.path.join(directory, label, filename)
    # print(aud_path)

    # audio, sample_rate = librosa.load(aud_path)
    # print(sample_rate)
    # print(audio)

    # print(get_dataset(label))

    # print(label+" example: ")
    # print(demo(filename, label))    

    #### --------------------------------------------------------------------------

    np.random.seed(42) 
    dataset_size = 400
    train_percentage = 0.8
    
    color_map = {'coughing': 'red', 'sneezing': 'blue', 'speech': 'green', 'silence': 'grey'}

    ## initialise the dataset
    dataset = {}
    for label in classes:
        data = get_dataset(label)
        np.random.shuffle(data)          # shuffle the order
        dataset[label] = data[:dataset_size]    # only take the fixed size

    ## split into train-test sets
    trainset, testset = split_dataset(dataset, train_percentage)

    ## train GMMs
    gmm = gmm_classifier(trainset,load_gmms=True,load_features=True,n_components=8)

    ## testing
    # y_gold = []
    # y_pred = []
    # for filename,label in testset:
    #     gold_label = label
    #     pred_label = gmm.classify(filename, label)
    #     y_gold.append(gold_label)
    #     y_pred.append(pred_label)
    # y_gold = np.array(y_gold)
    # y_pred = np.array(y_pred)

    ## evaluate the results
    # confusion, norm_confusion, accuracy, p, r, f1 = evaluate(y_gold, y_pred)
    # print("confusion matrix:\n",confusion)
    # print("accuracy: ",accuracy)
    # print("precision: ",p)
    # print("recall: ",r)
    # print("f1 score: ",f1)

    ## framewise prediction
    label = "mixed" # classes[1]
    filename = "qPo0bymnsh0.wav" #"_kwn1b3lq90_10.7-12.01.wav" #"_kwn1b3lq90_43.6-44.11.wav" #"_0rh6xgxhrq_9.18-15.85.wav"
    timestamps, pred_labels = gmm.classify_framewise(filename, label)
    # print(np.unique(pred_labels, return_counts=True))
    
    print("gold: ", label)
    # print(pred_labels)
    plotter = TimescalePlotter(list(timestamps), pred_labels, filename, label)
    plotter.plot()

    ## framewise testing & plotting
    # y_gold = []
    # y_pred = []
    # for filename,label in testset[:20]:
    #     timestamps, pred_labels = gmm.classify_framewise(filename, label)
    #     plotter = TimescalePlotter(list(timestamps), pred_labels, filename, label)
    #     plotter.plot()

