# Importing Required Libraries
import os 
import mne
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mne.channels import read_custom_montage
from mne import filter
from mne.filter import create_filter
import autoreject
import time 
####
matplotlib.use("TkAgg")
data_path = 'Specify Main Data Path'
locations = 'Specify path for location e:g biosemi.loc'

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


def get_edf_files(path):
    '''
    This function takes the path of the directory for the suject
    and gives the list of the files to be preprocessed we only 
    require files from file # 3 onwards and only the files with .edf
    extension
    '''

    files = os.listdir(path)
    consider = []
    not_consider = ['R01','R02','DS_Store']

    for file in files:
        extension = file.split('.')
        if 'DS_Store' in extension:
            extension.remove('DS_Store')
        if len(extension) == 2:
            if extension[0][4:] not in not_consider:
                if file not in consider:
                    consider.append(file)
    return consider

montage  = read_custom_montage(locations)


######################################
## Reading the data and Preprocessing
######################################
def preprocess_data(data_path, montage_data, rem_ocular_only = True, output_file_name = None):
    data = mne.io.read_raw_edf(data_path,preload=True)
    srate  = 160
    rename_dict = {}
    for i,j in zip(data.ch_names,montage.ch_names):
        if i!=j:
            print(i,j)
            rename_dict[i] = j
    data.rename_channels(rename_dict)
    data.set_montage(montage_data)
    # Mean Normalizing the data
    data = data.apply_function(lambda x: x - x.mean(), picks='all')
    print(60*'#')
    print(20*'#','Data Info',20*'#')

    ################### Filtering ######################
    data_ = data.get_data() 
    sfreq = srate

    # Applying the high-pass filter to remove frequencies below 0.5 Hz
    data_highpass = filter.filter_data(data_, sfreq, l_freq=0.5, h_freq=None, fir_design='firwin', phase='zero')

    # Apply the notch filter to remove frequencies between 59.5 and 60.5 Hz
    notch_freq = 60.0
    freqs = [60]  # Frequencies to be removed
    fir_design = 'firwin'  # Use FIR filter design
    phase = 'zero'  # Set to 'zero' for causal filter
    # Apply the band-stop filter to remove frequencies in the range of 59.5 to 60.5 Hz
    data_filtered = mne.filter.notch_filter(data_highpass, sfreq, freqs=freqs, fir_design=fir_design, phase=phase)


    # Create a new Raw object with the filtered data
    filtered_raw = mne.io.RawArray(data_filtered, data.info)

    ################################### ICA ######################################################################
    raw_ica  = filtered_raw.copy()

    random_state = 42   # ensures ICA is reproducable each time it's run
    ica_n_components = .99     # Specify n_components as a decimal to set % explained variance
    ica = mne.preprocessing.ICA(
    n_components=0.99, method="fastica", max_iter="auto", random_state=97)
    ica.fit(raw_ica)
    
    if rem_ocular_only:
        ica_z_thresh = 1.96
        eog_indices, eog_scores = ica.find_bads_eog(raw_ica, 
                                            ch_name=['Fp1.','Fp2.' ,'F8..','F7..'], 
                                            threshold=ica_z_thresh ,measure ='zscore')
        #ica.plot_scores(eog_scores, exclude=eog_indices)
        ica.exclude = eog_indices
    else:
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw_ica)
        ica_z_thresh = 1.96
        eog_indices, eog_scores = ica.find_bads_eog(raw_ica, 
                                            ch_name=['Fp1.','Fp2.' ,'F8..','F7..'], 
                                            threshold=ica_z_thresh ,measure ='zscore')
        ica.exclude = eog_indices + muscle_indices
    reconstructed_data = ica.apply(raw_ica)
    reconstructed_data.set_annotations(data.annotations)
    mne.export.export_raw(output_file_name,reconstructed_data)
subjects = ['S001', 'S008', 'S009', 'S014','S015', 'S016', 'S018', 'S019', 'S026', 'S033']
filter_directory = '/Users/sultm0a/Documents/Sipan Collaboration/Data/Filtered_Data_Ocular_Only'
isExist = os.path.exists(filter_directory)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(filter_directory)
   print("The new directory is created!")
for subject in subjects:
       

    subject_path = os.path.join(data_path,subject)
    considered_files = get_edf_files(subject_path)
    print(considered_files)
    print(60*'#')
    filter_subject_specific_path = os.path.join(filter_directory,subject)
    isExist = os.path.exists(filter_subject_specific_path)
    if not isExist:
   # Create a new directory because it does not exist
        os.makedirs(filter_subject_specific_path)
    for file_ in considered_files:
        output_file_name = os.path.join(filter_subject_specific_path,file_)
        input_file_name = os.path.join(subject_path,file_)
        print('Working on File {}'.format(input_file_name))
        preprocess_data(data_path = input_file_name , montage_data = montage, rem_ocular_only = True, output_file_name = output_file_name )





