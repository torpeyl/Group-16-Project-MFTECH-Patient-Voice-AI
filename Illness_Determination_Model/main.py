from .wav_to_spec import wav_to_spec
from .CNN_Illness_Classifier import illness_classifier_CNN

def illness_classifier(wav_file_path):

    spectro_file_path = wav_to_spec(wav_file_path)
    print('.wav to .jpg successful.')
    print(spectro_file_path)
    result = illness_classifier_CNN(spectro_file_path)
    
    return result
