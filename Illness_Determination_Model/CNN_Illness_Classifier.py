import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import load_model

def illness_classifier_CNN(spectro_file_path):
    
    image = cv2.imread(spectro_file_path)
    print('Reading image successful.')
    image = cv2.resize(image, (264, 264))
    print('Resizing image successful.')
    image = image.astype('float32')
    print('Image converted to 32 bit float.')
    image /= .255
    print('Image pixel values scaled between 0-1 successfully.')
    image = np.expand_dims(image, axis=0)

    CNN = load_model('./Illness_Determination_Model/Liam_Model.h5')
    print('CNN loaded successfully.')
    
    prediction = CNN.predict(image)
    print('Model predicted successfully.')
    
    #0 means healthy, 1 means ill
    return f'{prediction[0][0]}'


