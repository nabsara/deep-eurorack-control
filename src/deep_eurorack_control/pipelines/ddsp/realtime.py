

import torch
import os
os.chdir(r"C:\Users\NILS\Documents\ATIAM\PAM\deep-eurorack-control\src")

import time
from deep_eurorack_control.models.ddsp.model import DDSP
from deep_eurorack_control.models.ddsp.decoder import Decoder
from deep_eurorack_control.config import settings
from deep_eurorack_control.models.ddsp.ops import *


logdir = r"C:\Users\NILS\Documents\ATIAM\PAM\deep-eurorack-control\data\save"




sr = 16000
frame_size= 256
n_harmonics = 51
n_bands = 65


# def load_model(checkpoint_path,sr,frame_size,n_harmonics,n_bands):
#     checkpoint = torch.load(checkpoint_path,map_location=settings.device)
#     model = DDSP(sr,frame_size,n_harmonics,n_bands,residual=False)
#     model.decoder.load_state_dict(checkpoint["model_state_dict"])
#     return(model)

def load_model(checkpoint_path,sr,frame_size,n_harmonics,n_bands):
    checkpoint = torch.load(checkpoint_path,map_location=settings.device)
    # model = DDSP(sr,frame_size,n_harmonics,n_bands,residual=False)
    decoder = Decoder(sr,n_harmonics,n_bands)
    # decoder = decoder.load_state_dict(checkpoint["model_state_dict"])
    return(decoder)



# if __name__ =="__main__":

model_name = r"DDSP_lr_{lr}_n_epochs_{n_epochs}__sr_{self.sr}__frame_{self.frame_size}.pt"
checkpoint_path = os.path.join(logdir,model_name)
model = load_model(checkpoint_path,sr,frame_size,n_harmonics,n_bands)
x = torch.rand(1,1,1).to(settings.device)
init_phase = torch.tensor([0]).to(settings.device)
i=0
while i<10:
    time.time()
    pitch = x
    loud = x
    harmonics,filters = model.decoder.real_time_forward(pitch,loud)
    signal_out,final_phase = generate_signal(pitch,harmonics,filters,model.frame_size,model.sr,init_phase)
    init_phase = final_phase
    print(signal_out.shape)
    i+=1

    
    
    