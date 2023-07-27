# Importing Required Libraries
import os 
import mne
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mne.channels import read_custom_montage
from mne.filter import create_filter
import autoreject
import time 
####
matplotlib.use("TkAgg")
data_path = '/Users/sultm0a/Documents/Sipan Collaboration/Data/files'
locations = '/Users/sultm0a/Documents/Sipan Collaboration/Data/files/BioSemi64.loc'

subjects = ['S038', 'S007', 'S031', 'S009', 'S036', 'S096', 'S062', 
            'S065', 'S091', 'S053', 'S098', 'S054', 'S008', 'S037', 'S030', 'S039', 
            'S006', 'S001', 'S055', 'S052', 'S099', 'S090', 'S064', 'S063', 'S097', 
            'S041', 'S079', 'S046', 'S084', 'S070', 'S048', 'S077', 'S083', 'S023', 'S024', 
            'S012', 'S015', 'S049', 'S082', 'S076', 'S071', 'S085', 'S078', 'S047', 
            'S040', 'S014', 'S013', 'S025', 'S022', 'S103', 'S104', 'S105', 'S102', 
            'S066', 'S092', 'S059', 'S095', 'S061', 'S057', 'S068', 'S050', 
            'S004', 'S003', 'S035', 'S032', 'S051', 'S056', 'S069', 'S060', 'S094', 'S093', 'S067', 'S058', 
            'S033', 'S034', 'S002', 'S005', 'S027', 'S018', 'S020','S016', 'S029', 
            'S011', 'S045', 'S089', 'S042', 'S074', 'S080', 'S087', 'S073', 'S010', 'S017', 
            'S028', 'S021', 'S026', 'S019', 'S072', 'S086', 'S081', 'S075', 'S088', 'S043', 'S044', 
            'S109', 'S107', 'S100', 'S101', 'S106', 'S108']

example_subject = 'S008'

subject_path = os.path.join(data_path,example_subject)

subject_files = os.listdir(subject_path)
def get_edf_files(path):
    '''
    This function takes the path of the directory for the suject
    and gives the list of the files to be preprocessed we only 
    require files from file # 3 onwards and only the files with .edf
    extension
    '''

    files = os.listdir(path)
    consider = []
    not_consider = ['R01','R02']

    for file in files:
        extension = file.split('.')
        if len(extension) == 2:
            if extension[0][4:] not in not_consider:
                if file not in consider:
                    consider.append(file)
    return consider


consider  = get_edf_files(subject_path)
print(60*'#')
print('Considered')
print(consider)
print(60*'#')
montage  = read_custom_montage(locations)
plt.ioff()
montage.plot()
plt.show(block=False)
time.sleep(5)
print('Script Running')

######################################
## Reading the data and Preprocessing
######################################
def preprocess_data(data_path, montage_data):
    pass
