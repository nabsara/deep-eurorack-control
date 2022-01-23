import os

os.chdir(r'C:\Users\NILS\Documents\ATIAM\PAM\deep-eurorack-control\src')

from deep_eurorack_control.pipelines.ddsp.dataset_process import preprocess_dataset


raw_data_dir = r'C:\Users\NILS\Documents\ATIAM\PAM\Datasets\nsynth-test\audio'
dataset_dir = r'C:\Users\NILS\Documents\ATIAM\PAM\Datasets\strings_test'
filters = ['string_acoustic']
sr = 16000
frame_size = 256
nb_files = None

preprocess_dataset(raw_data_dir,dataset_dir,filters,sr,frame_size,nb_files)
