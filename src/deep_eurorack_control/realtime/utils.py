import torch 
from deep_eurorack_control.models.ddsp.model import DDSP

import numpy as np

class settings:
    def __init__(self):
        self.device = 'cpu'

settings = settings()


def load_model(checkpoint_path,sr,frame_size,n_harmonics,n_bands):
    checkpoint = torch.load(checkpoint_path,map_location=settings.device)
    model = DDSP(sr,frame_size,n_harmonics,n_bands,residual=False)
    model.decoder.load_state_dict(checkpoint["model_state_dict"])
    return(model)



def generate_signal_realtime(pitch,harmonics,filters,frame_size,sr,init_phase=0):
    amps = harmonics[:,:,1:]
    level = harmonics[:,:,:1]
    freqs = pitch*torch.arange(1,amps.shape[-1]+1).to(settings.device)[None,None,:]
    amps = amps*((freqs<sr/2).float()+1e-4)
    amps = level*amps/torch.sum(amps,axis=-1,keepdim=True)
    len_signal = pitch.shape[1]*frame_size
    amps = upsample(amps,len_signal)
    f0  =  upsample(pitch,len_signal)
    noise = noise_synth(filters,frame_size)
    harmo,final_phase = h_synth(f0,amps,sr,init_phase)
    return(harmo+noise,final_phase)


def h_synth(f0,amps,sr,init_phase=0):    
    phases = torch.cumsum(2 * np.pi * f0 / sr, 1) + init_phase
    final_phase = phases[-1]
    phases = phases * torch.arange(1,amps.shape[-1]+1).to(settings.device)[None,None,:]
    wave = amps*torch.sin(phases)
    return(torch.sum(wave,axis=-1),final_phase)

    


def noise_synth(H,frame_size):
    H = torch.stack([H,torch.zeros_like(H)],-1)
    H = torch.view_as_complex(H)
    
    h = irfft(H)
    h = torch.roll(h,int(h.shape[-1]//2))
    size_dif = frame_size-h.shape[-1]
    h = torch.nn.functional.pad(h,(int(size_dif/2),int(size_dif/2)))
    w = torch.hann_window(frame_size).to(settings.device)
    h = w[None,None,:]*h
    h = torch.roll(h,-int(h.shape[-1]//2))
    H = rfft(h)
    
    noise =2*torch.rand(H.shape[0],H.shape[1],frame_size).to(settings.device)-1
    
    noise_fft = rfft(noise)
    noise_out = irfft(noise_fft * H)

    noise_out = noise_out.reshape(noise_out.shape[0],-1)
    return (noise_out)


def upsample(array,n_final):
    array_temp =  torch.nn.functional.interpolate(array.permute(0,2,1),n_final,mode='linear')
    return(array_temp.permute(0,2,1))



def rfft(x):
    x = x.cpu().numpy()
    x = np.fft.rfft(x)
    x = torch.from_numpy(x)
    return x


def irfft(x):
    x = x.cpu().numpy()
    # x = x[..., 0] + 1j * x[..., 1]
    x = np.fft.irfft(x)
    x = torch.from_numpy(x)
    return x
