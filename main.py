"""
This program processes all of the info.txt files and prints all of the different values in the data for each data field.
The number to the right of each data value is the number of times it occurs in the dataset.
In order to use change the directory to the location of your downloaded data.
"""

import os
import re
from colorama import Fore, Style

directory = 'data'

file_names = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        if f[-8:] == 'info.txt':
            file_names.append(f[len(directory) + 1:])

info_dict = {}
for file in file_names:
    with open(os.path.join(directory, file), 'r') as f:
        lines = f.readlines()
        condition = lambda line: re.sub(r'[\t]', ':', line) if ':' not in line else line
        cleaned_lines = [condition(line) for line in lines]
        cleaned_lines = [re.sub(r'[\n\t]', '', s) for s in cleaned_lines]
        cleaned_lines= [item for item in cleaned_lines if item != ':']
        for pair in cleaned_lines:
            li = pair.split(':')
            # check if key exists, if not create set for that and add first value
            if li[0] in info_dict.keys():
                if li[1] in info_dict[li[0]].keys():
                    info_dict[li[0]][li[1]] += 1
                else:
                    info_dict[li[0]][li[1]] = 1
            else:
                info_dict[li[0]] = {li[1]: 1}


keys = ['ID', 'Age']        # NOTE THAT THE DATA FIELDS IN THIS LIST WILL NOT BE PRINTED TO 
                            # THE TERMINAL, HOWEVER THEY ARE STILL PRESENT IN THE DATA 
for key in info_dict.keys():
    if key not in keys:
        print(Fore.BLUE + key + Style.RESET_ALL)
        print(info_dict[key])

# print({x: info_dict[x] for x in info_dict if x not in keys})