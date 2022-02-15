import torch
from tqdm import tqdm
import time
import os
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from deep_eurorack_control.config import settings

from deep_eurorack_control.models.ddsp.decoder import Decoder
from deep_eurorack_control.models.ddsp.encoder import Encoder

from deep_eurorack_control.models.ddsp.ops import *
from deep_eurorack_control.helpers.ddsp import plot_metrics

class DDSP:
    def __init__(self,sr,frame_size,n_harmonics,n_bands,residual=False,n_z=16):

        self.sr = sr
        self.frame_size = frame_size
        self.n_harmonics = n_harmonics 
        self.n_bands = n_bands
        self.scales = [2048,1024,512,256,128,64]
        self.residual = residual
        self.n_z = n_z
        self.n_mfcc = 30
        
        
        self.decoder = Decoder(self.sr,self.n_harmonics,self.n_bands,self.residual,self.n_z).to(settings.device)
        
        if self.residual==True:
            self.encoder = Encoder(self.n_z,self.n_mfcc).to(settings.device)
            self.wave2mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=self.n_mfcc, melkwargs={"n_fft": 1024,"hop_length":int(1024/4),"f_min":20,"f_max":int(sr/2),"n_mels":128,"center":True}).to(settings.device)
        
    def _init_optimizer(self, learning_rate,alpha):
        if self.residual==True:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        else:
            params =self.decoder.parameters()
            
        self._opt = torch.optim.Adam(
            params, lr=learning_rate)
        schedule = self._init_schedule(alpha)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self._opt, schedule)
    
    def _compute_loss_spec(self,signal_in,signal_out,alpha=1):
        loss_batch = spectral_loss(self.scales,signal_in,signal_out,alpha)
        return(torch.mean(loss_batch))
    
    def _init_schedule(self,alpha):
        def schedule(epoch):
          i = epoch//5
          print(alpha**i)
          return (alpha**i) 
        return schedule
    
    
    def train(self,dataloader,lr,n_epochs,display_step,logdir,alpha=0.98):
        self._init_optimizer(lr,alpha)
                
        start = time.time()
    
        writer = SummaryWriter(
            os.path.join(
                logdir,
                f"TB/DDSP_lr_{lr}_n_epochs_{n_epochs}__sr_{self.sr}"
                + time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()),
            )
        )
        
        losses=[]
        display_loss =0
        it_display = 0
        print(settings.device)

        it = 0  # number of batch iterations updated at the end of the dataloader for loop
        for epoch in range(n_epochs):
            for data in tqdm(dataloader):
                pitch_true,pitch_conf,loud,audio = [item.to(settings.device) for item in data]
                signal_in = audio.reshape(audio.shape[0],-1)
                
                self._opt.zero_grad()
            
                if self.residual==True:
                    mfcc = self.wave2mfcc(signal_in)[...,:-1]
                    res = self.encoder(mfcc.permute(0,2,1))
                else:
                    res=None

                pitch = pitch_true

                harmonics,filters = self.decoder(pitch,loud,res)
                
                signal_out,_ = generate_signal(pitch,harmonics,filters,self.frame_size,self.sr)
                
                spec_loss = self._compute_loss_spec(signal_in,signal_out)
                loss = spec_loss
                loss.backward()
              
                self._opt.step()
                losses.append(loss.item())
                display_loss += loss.item()
                it_display+=1 
                it+=1
                
                
                if (it-1) % display_step == 0: #or (it== len(dataloader) - 1):
                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{it}/{len(dataloader)}]"
                        f"\tTime: {time.time() - start} (s)\tLoss: {display_loss/it_display}"
                    )
                    
                    writer.add_scalar(
                            "Total Loss",
                            display_loss / it_display,
                            epoch * len(dataloader) + it,
                        )                            
                                        
                    display_loss = 0 
                    it_display = 0 
                    
                    with torch.no_grad():
                        real_audio = signal_in.detach().cpu()
                        rec_audio = signal_out.detach().cpu()
                        rec_harmonics = harmonics.detach().cpu().numpy()
                        rec_filters = filters.detach().cpu().numpy()
                        pitch_true = pitch_true.detach().cpu().numpy()

                        for j in range(2):
                            figure = plot_metrics(pitch_true[j],real_audio[j].numpy(),rec_audio[j].numpy(),rec_harmonics[j],rec_filters[j],self.sr,self.frame_size)
                            writer.add_audio(
                                        "Reconstructed Sounds/" + str(j),
                                        rec_audio[j],
                                        global_step=epoch * len(dataloader) + it,
                                        sample_rate=self.sr,
                                    )
                                                    
                            writer.add_audio(
                                        "Real Sounds/" + str(j),
                                        real_audio[j],
                                        global_step=epoch * len(dataloader) + it,
                                        sample_rate=self.sr,
                                    )   
                            
                            writer.add_figure(
                            "Output Images/" + str(j),
                            figure,
                            global_step=epoch * len(dataloader) + it,
                            )   
            self.scheduler.step()            
            if epoch % 10 == 0 or epoch == n_epochs - 1 :                
                torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.decoder.state_dict(),
                    "optimizer_state_dict": self._opt.state_dict(),
                    "loss": losses,
                }, os.path.join(logdir, "DDSP_lr_{lr}_n_epochs_{n_epochs}__sr_{self.sr}__frame_{self.frame_size}.pt"))