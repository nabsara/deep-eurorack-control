from warnings import resetwarnings
import torch.nn as nn 
import torch

class Encoder(nn.Module):
    def __init__(self,n_z, n_mfcc,n_hidden=512):
        super().__init__()
        
        self.norm = nn.LayerNorm(n_mfcc)
        self.gru = nn.GRU(n_mfcc,n_hidden)
        self.dense = nn.Linear(n_hidden,n_z)
        
    
    def forward(self, mfcc):
        temp = self.norm(mfcc)
        temp,_ = self.gru(temp)
        return (self.dense(temp))