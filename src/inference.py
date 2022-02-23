
import os
# os.chdir(r'C:\Users\NILS\Documents\ATIAM\PAM\deep-eurorack-control\src')

import torch
import time
import numpy as np
import argparse
import timeit
import samplerate
import cdpam

from tqdm import tqdm
from deep_eurorack_control.models.ddsp.model import DDSP
from deep_eurorack_control.models.ddsp.decoder import Decoder
from deep_eurorack_control.config import settings
from deep_eurorack_control.models.ddsp.ops import *
from deep_eurorack_control.datasets.ddsp import  NSynth_ddsp
from torch.utils.data import DataLoader
import torchaudio

def load_model(filepath,filename,n_hidden):
    checkpoint_file = os.path.join(filepath,filename)
    checkpoint = torch.load(checkpoint_file,map_location=settings.device)
    sr,frame_size,n_harmonics,n_bands = checkpoint["sr"],checkpoint["frame_size"], checkpoint["n_harmonics"],checkpoint["n_bans"]
    model_name = '_'.join(checkpoint_file.split('lr')[0].split('_')[1:-1])
    if 'res' in model_name:
        residual=False
    else:
        residual=True
    print(model_name,sr,frame_size,n_harmonics,n_bands,residual)
    model = DDSP(model_name,sr,frame_size,n_harmonics,n_bands,n_hidden,residual=residual)
    model.decoder.load_state_dict(checkpoint["model_state_dict"])
    return model

def compute_loss(dataloader,model,loss_fn,loss):
    
    model.decoder.eval()
    torch.set_grad_enabled(False)

    multiscale_loss = 0 
    jnd_loss = 0
    mel_loss=0
    start=time.time()
    for data in tqdm(dataloader):
 
        pitch_true,pitch_conf,loud,audio = [item.to(settings.device) for item in data]
        signal_in = audio.reshape(audio.shape[0],-1)

        if model.residual==True:
            mfcc = model.wave2mfcc(signal_in)[...,:-1]
            res = model.encoder(mfcc.permute(0,2,1))
        else:
            res=None

        pitch = pitch_true
        harmonics,filters = model.decoder(pitch,loud,res)
        signal_out,_ = generate_signal(pitch,harmonics,filters,model.frame_size,model.sr)
        
        if loss==True:
            multiscale_loss+= model._compute_loss_spec(signal_in,signal_out).item()/len(dataloader)
            mel_loss += torch.mean(melspectrogram_loss(signal_in,signal_out)).item()/len(dataloader)
            
            
            resample_rate = 22050
            resampler = torchaudio.transforms.Resample(model.sr, resample_rate,dtype=signal_out.dtype).to(settings.device)
            jnd_loss += torch.mean(loss_fn.forward(resampler(signal_in)*32768,resampler(signal_out)*32768)).item()/len(dataloader)

    if loss==True:
        return(multiscale_loss,mel_loss,jnd_loss)
    else:
        end = time.time()
        total_time=(end-start)
        return(total_time)
    

def run_analysis(filepath,filename,n_hidden,dataloader,batch_size):
    model = load_model(filepath,filename,n_hidden)
    loss_fn = cdpam.CDPAM(dev=settings.device)
    inf_time = 0
    for i in range(10):
        inf_time += compute_loss(dataloader,model,loss_fn,loss=False)/10
    inf_speed =  64000*len(dataloader)*batch_size/inf_time
    # multiscale_loss,mel_loss,jnd_loss = compute_loss(dataloader,model,loss_fn,loss=True)
    print('Model : ',filename)
    # print("Multiscale Spectral Loss : ",round(multiscale_loss,2))
    # print("Mel Loss : ",round(mel_loss,2))
    # print("JND Loss : ",round(jnd_loss,2))
    print("Inference Speed : ",round(inf_speed*1e-6,2)," M samples/sec")
    # return(inf_speed,multiscale_loss,mel_loss,jnd_loss)





from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
mel_basis = {}
hann_window = {}

n_fft = 1024
num_mels = 80
sampling_rate = 16000
hop_size = 256
win_size = 1024
fmin = 0
fmax = 8000

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(x, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    #if torch.min(x) < -1.:
    #    print('min value is ', torch.min(x))
    #if torch.max(x) > 1.:
    #    print('max value is ', torch.max(x))

    global mel_basis, hann_window, device
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(x.device)] = torch.from_numpy(mel).float().to(x.device)
        hann_window[str(x.device)] = torch.hann_window(win_size).to(x.device)

    
    x = torch.nn.functional.pad(x.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    x = x.squeeze(1)

    melspectro = torch.stft(x, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(x.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    melspectro = torch.sqrt(melspectro.pow(2).sum(-1)+(1e-9))

    melspectro = torch.matmul(mel_basis[str(fmax)+'_'+str(x.device)], melspectro)
    melspectro = spectral_normalize_torch(melspectro)

    return melspectro

# def melspectrogram_loss(x, x_gen, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
def melspectrogram_loss(x, x_gen):
    x_mel = mel_spectrogram(x, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True)
    x_gen_mel = mel_spectrogram(x_gen, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True)
    loss_melspectro = F.l1_loss(x_mel, x_gen_mel)
    return loss_melspectro






if __name__ =="__main__":
    # filepath = r"C:\Users\NILS\Documents\ATIAM\PAM\Nouveau dossier" 
    filepath = "/net/homes/u.atiam/u.atiam21-22/demerle/test/deep-eurorack-control/data/Nouveau dossier" 
    

    filenames = os.listdir(filepath)

    # dataset_dir = r'C:\Users\NILS\Documents\ATIAM\PAM\Datasets\nsynth_test_processed'
    dataset_dir = '/fast-1/atiam/nils/nsynth_test_processed'

    batch_size =1
    dataset = NSynth_ddsp(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for filename in filenames:
        if 'gru' in filename:
            n_hidden = 256
        else:
            n_hidden=512
        print(n_hidden)
        run_analysis(filepath,filename,n_hidden,dataloader,batch_size)


    
    