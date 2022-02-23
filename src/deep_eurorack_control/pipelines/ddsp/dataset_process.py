import os 
import shutil
from tqdm import tqdm
import numpy as np
import librosa

from deep_eurorack_control.helpers.utils import save_pickle
from deep_eurorack_control.models.ddsp.ops import get_loudness,get_pitch


def preprocess_dataset(raw_data_dir,dataset_dir,filter,sr,frame_size,nb_files=None):
    files= os.listdir(raw_data_dir) 
    os.makedirs(dataset_dir,exist_ok=True)
    audio_files = []
    for file in files:
            if filter in file:
                audio_files.append(os.path.join(raw_data_dir,file))
                # break
    
        
    if nb_files is not None:
        audio_files=audio_files[:nb_files]
    
    nb_samples = len(audio_files)
    nb_trames = int(len(librosa.load(audio_files[0],sr)[0])/frame_size)
    
    loudness_arr = np.zeros((nb_samples,nb_trames,1))
    pitch_arr = np.zeros((nb_samples,nb_trames,1))
    pitch_conf_arr = np.zeros((nb_samples,nb_trames,1))

    audio_arr = np.zeros((nb_samples,nb_trames,frame_size))
    
    
    for i,file in enumerate(tqdm(audio_files)):
            signal,_ = librosa.load(file,sr)
            pitch,pitch_conf = get_pitch(signal,sr,frame_size)
            pitch_arr[i] = pitch.reshape(-1,1)
            pitch_conf_arr[i] = pitch_conf.reshape(-1,1)
            loudness_arr[i] = get_loudness(signal,sr,frame_size,n_fft=1024).reshape(-1,1)
            audio_arr[i] = signal.reshape(-1,frame_size)
            
        
        
    l_mean,l_std = np.mean(loudness_arr),np.std(loudness_arr)
    loudness_arr = (loudness_arr-l_mean)/l_std
            
    save_pickle(pitch_arr,os.path.join(dataset_dir,'pitch.pkl'))
    save_pickle(pitch_conf_arr,os.path.join(dataset_dir,'pitch_conf.pkl'))
    save_pickle(loudness_arr,os.path.join(dataset_dir,'loudness.pkl'))
    save_pickle(audio_arr,os.path.join(dataset_dir,'audio.pkl'))
    
    
def preprocess_dataset_violin(raw_data_dir,dataset_dir,sr,frame_size,len_samples=4):
    files= os.listdir(raw_data_dir)
    audio_size = len_samples*sr
    os.makedirs(dataset_dir,exist_ok=True)
    
    audio_data = np.empty((0,audio_size))
    print(audio_data.shape)
    print('Reading Files')
    for file in tqdm(files):
        audio_file = os.path.join(raw_data_dir,file)
        signal,_ = librosa.load(audio_file,sr)
        
        nb_subfiles = signal.shape[0]//audio_size+1
        signal = np.pad(signal,(0,nb_subfiles*audio_size-signal.shape[0]))
        signal = signal.reshape(-1,audio_size)
        print(signal.shape)
        audio_data = np.row_stack((audio_data,signal))
    
    
    print(audio_data.shape)
    nb_samples = audio_data.shape[0]
    nb_trames = int(audio_size/frame_size)
    
    loudness_arr = np.zeros((nb_samples,nb_trames,1))
    pitch_arr = np.zeros((nb_samples,nb_trames,1))
    pitch_conf_arr = np.zeros((nb_samples,nb_trames,1))

    audio_arr = np.zeros((nb_samples,nb_trames,frame_size))
    
    print('Extracting Pitch and Loudness')

    for i,signal in enumerate(tqdm(audio_data)):
            pitch,pitch_conf = get_pitch(signal,sr,frame_size)
            pitch_arr[i] = pitch.reshape(-1,1)
            pitch_conf_arr[i] = pitch_conf.reshape(-1,1)
            loudness_arr[i] = get_loudness(signal,sr,frame_size,n_fft=1024).reshape(-1,1)
            audio_arr[i] = signal.reshape(-1,frame_size)
            
        
        
    l_mean,l_std = np.mean(loudness_arr),np.std(loudness_arr)
    loudness_arr = (loudness_arr-l_mean)/l_std
            
    save_pickle(pitch_arr,os.path.join(dataset_dir,'pitch.pkl'))
    save_pickle(pitch_conf_arr,os.path.join(dataset_dir,'pitch_conf.pkl'))
    save_pickle(loudness_arr,os.path.join(dataset_dir,'loudness.pkl'))
    save_pickle(audio_arr,os.path.join(dataset_dir,'audio.pkl'))