from warnings import resetwarnings
import torch.nn as nn 
import torch

class Decoder(nn.Module):
    def __init__(self, n_amps,n_bands,residual=False,n_z=16,n_hidden=512):
        super().__init__()

        self.residual=residual

        self.mlp_pitch = self.make_mlp(1,n_hidden)
        self.mlp_loud = self.make_mlp(1,n_hidden)
        
        if self.residual==True:
            self.mlp_res = self.make_mlp(n_z,n_hidden)
            self.gru = nn.GRU(n_hidden*3,n_hidden)
        else:
            self.gru = nn.GRU(n_hidden*2,n_hidden)
        
        self.mlp_out = self.make_mlp(n_hidden+2,n_hidden)
        
        self.dense_amps = nn.Linear(n_hidden,n_amps)
        self.dense_filters = nn.Linear(n_hidden,n_bands)
        self.dense_f0 = nn.Linear(n_hidden,1)

    def make_mlp(self, n_input, n_hidden):
        def make_layer(n_in,n_out):
            return(nn.Sequential(
                nn.Linear(n_in,n_out),
                nn.LayerNorm(n_out),
                nn.LeakyReLU(inplace=True)
            ))
                
        return(nn.Sequential(
            make_layer(n_input,n_hidden),
            make_layer(n_hidden,n_hidden),
            make_layer(n_hidden,n_hidden)
        ))
    
    def forward(self, pitch,loud,res=None):
        pitch_out = self.mlp_pitch(pitch)
        loud_out  = self.mlp_loud(loud)
        if self.residual == True:
            res_out = self.mlp_res(res)
            gru_out,_  = self.gru(torch.cat([pitch_out,loud_out,res_out],-1))
        else:
            gru_out,_  = self.gru(torch.cat([pitch_out,loud_out],-1))
        
        net_out =self.mlp_out(torch.cat([gru_out,pitch,loud],-1))
        
        harmonics = self.dense_amps(net_out)
        harmonics = 2*torch.sigmoid(harmonics)**2.3025851 + 1e-7
        filters = self.dense_filters(net_out)
        filters = 2*torch.sigmoid(filters)**2.3025851 + 1e-7
        f0 = pitch*(1+torch.tanh(self.dense_f0(net_out)))
        return harmonics,filters,f0