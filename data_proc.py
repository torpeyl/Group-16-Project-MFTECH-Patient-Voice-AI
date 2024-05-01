# from IPython.display import display
import matplotlib.pyplot as plt
# matplotlib inline
import numpy as np
import os
import shutil
import posixpath

import wfdb

## Read a WFDB record using the 'rdrecord' function into a wfdb.Record object.
record = wfdb.rdrecord('data/voice001') 

record_attributes = record.__dict__

## Print the data
print(record_attributes["p_signal"])

## Plot the signals
wfdb.plot_wfdb(record=record, title='voice001')