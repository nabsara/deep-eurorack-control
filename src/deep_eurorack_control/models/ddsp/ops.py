import librosa 
import crepe
import numpy as np
import torch
from deep_eurorack_control.config import settings

def get_pitch(signal,sr,frame_size):
    step_size = int(1000*frame_size/sr)
    pitch = crepe.predict(signal,sr,viterbi=True,verbose=False,step_size=step_size)[1][:-1]
    return(pitch)

def get_loudness(signal,sr,frame_size,n_fft):
    stft = librosa.stft(signal,hop_length=frame_size,n_fft=n_fft,win_length=1024)[:,:-1]
    # freqs = np.linspace(0,sr/2,int(n_fft/2+1))
    freqs = librosa.fft_frequencies(sr, n_fft=n_fft)
    freqs[0] += 1e-6
    a_weighting = librosa.A_weighting(freqs)
    stft_log = np.log(np.abs(stft)+1e-7) + a_weighting.reshape(-1,1)
    loudness = np.mean(stft_log,axis=0)
    return(loudness)


def h_synth(f0,amps,sr):    
    phases = torch.cumsum(2 * np.pi * f0 / sr, 1)
    phases = phases * torch.arange(1,amps.shape[-1]+1).to(settings.device)[None,None,:]
    # phases = 2*np.pi*freqs/sr * torch.arange(0,amps.shape[1]).to(settings.device)[None,:,None]
    wave = amps*torch.sin(phases)
    return(torch.sum(wave,axis=-1))


def noise_synth(H,frame_size):
    H = torch.stack([H,torch.zeros_like(H)],-1)
    H = torch.view_as_complex(H)
    h = torch.fft.irfft(H)
    h = torch.roll(h,int(h.shape[-1]//2))
    size_dif = frame_size-h.shape[-1]
    h = torch.nn.functional.pad(h,(int(size_dif/2),int(size_dif/2)))
    w = torch.hann_window(frame_size).to(settings.device)
    h = w[None,None,:]*h
    h = torch.roll(h,-int(h.shape[-1]//2))
    H = torch.fft.rfft(h)
    
    noise =2*torch.rand(H.shape[0],H.shape[1],frame_size).to(settings.device)-1
    
    noise_fft = torch.fft.rfft(noise)
    noise_out = torch.fft.irfft(noise_fft * H)

    noise_out = noise_out.reshape(noise_out.shape[0],-1)
    return (noise_out)

def spectral_loss(scales,xin,xout,alpha=1):
    L_total = torch.zeros(xin.shape[0]).to(settings.device)
    for scale in scales:
        stft_in =  torch.abs(  torch.stft(xin,n_fft = scale,return_complex=True,normalized=True,window=torch.hann_window(scale).to(xin)    )  ) 
        stft_out = torch.abs(  torch.stft(xout,n_fft = scale,return_complex=True,normalized=True,window=torch.hann_window(scale).to(xout) )  )
        L_total += torch.mean(torch.abs(stft_in-stft_out),dim=(1,2)) + torch.mean(torch.abs(torch.log(stft_in+1e-5) - torch.log(stft_out+1e-5)),dim=(1,2))
        # L_total += torch.norm(stft_in-stft_out,1,dim = (1,2)) + alpha*torch.norm(torch.log(stft_in+1e-6) - torch.log(stft_out+1e-6) ,1,dim = (1,2))
    return(L_total)

def upsample(array,n_final):
    array_temp =  torch.nn.functional.interpolate(array.permute(0,2,1),n_final,mode='linear')
    return(array_temp.permute(0,2,1))

def smooth(params,frame_size,final_size):
    hop = int(frame_size/4)
    winlen = int(frame_size/2)
    
    batch_size = params.shape[0]
    nb_features = params.shape[-1]

    temp = params.permute(0,2,1)

    window = torch.hann_window(winlen).to(settings.device)
    window = window/(window[0] + window[int(winlen/2)])



    temp = temp.unfold(-1,winlen,hop)[:,:,:,int(winlen/2)][:,:,:,None]*window[None,None,None,:]
    temp =temp.reshape(int(nb_features*batch_size),-1,winlen).permute(0,2,1)

    temp = torch.nn.functional.fold(temp,output_size=(1,final_size),kernel_size=(1,winlen),stride=(1,hop)).squeeze().reshape(batch_size,-1,64000)
    return(temp.permute(0,2,1))
    


def generate_signal(pitch,harmonics,filters,frame_size,sr):
    amps = harmonics[:,:,1:]
    level = harmonics[:,:,:1]
    
    freqs = pitch*torch.arange(1,amps.shape[-1]+1).to(settings.device)[None,None,:]
    amps = amps*((freqs<sr/2).float()+1e-4)
    
    amps = level*amps/torch.sum(amps,axis=-1,keepdim=True)


    
    len_signal = pitch.shape[1]*frame_size
    
    amps = upsample(amps,len_signal)
    
    # amps = smooth(amps,frame_size,len_signal)
    
    f0  =  upsample(pitch,len_signal)
    # f0 = smooth(f0,frame_size,len_signal)

    signal = h_synth(f0,amps,sr)+ #noise_synth(filters,frame_size)
    return(signal)


