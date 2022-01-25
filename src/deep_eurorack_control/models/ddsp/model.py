import torch
from tqdm import tqdm
import time
import os
from torch.utils.tensorboard import SummaryWriter
from deep_eurorack_control.config import settings

from deep_eurorack_control.models.ddsp.decoder import Decoder
from deep_eurorack_control.models.ddsp.ops import *
from deep_eurorack_control.helpers.ddsp import plot_metrics


class DDSP:
    def __init__(self,sr,frame_size,n_harmonics,n_bands):

        self.sr = sr
        self.frame_size = frame_size
        self.n_harmonics = n_harmonics 
        self.n_bands = n_bands
        self.scales = [2048,1024,512,256,128,64]
        
        self.decoder = Decoder(self.n_harmonics,self.n_bands).to(settings.device)
        
        
    def _init_optimizer(self, learning_rate,beta_1=0.9,beta_2=0.999):
        self._opt = torch.optim.Adam(
            self.decoder.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )
    
    def _compute_loss(self,signal_in,signal_out,alpha=1):
        loss_batch = spectral_loss(self.scales,signal_in,signal_out,alpha)
        return(torch.mean(loss_batch))
    
    
    def train(self,dataloader,lr,n_epochs,display_step,logdir):
        self._init_optimizer(lr)
                
        start = time.time()
    
        writer = SummaryWriter(
            os.path.join(
                logdir,
                f"TB/DDSP_lr_{lr}_n_epochs_{n_epochs}__sr_{self.sr}"
                + time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()),
            )
        )
        
        losses=[]
        display_loss = 0
        it_display = 0
        
        it = 0  # number of batch iterations updated at the end of the dataloader for loop
        for epoch in range(n_epochs):
            it=0
            for data in tqdm(dataloader):
                pitch,loud,audio = data
                pitch = pitch.to(settings.device)
                loud = loud.to(settings.device)
                signal_in = audio.reshape(audio.shape[0],-1).to(settings.device)
                
                self._opt.zero_grad()
                harmonics,filters = self.decoder(pitch,loud)

                signal_out = generate_signal(pitch,harmonics,filters,self.frame_size,self.sr)
                
                loss = self._compute_loss(signal_in,signal_out)
                loss.backward()
              
                self._opt.step()
                losses.append(loss.item())
                display_loss += loss.item()
                it_display+=1 
                it+=1
                
                
                if (it-1) % display_step == 0 or (it== len(dataloader) - 1):
                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{it}/{len(dataloader)}]"
                        f"\tTime: {time.time() - start} (s)\tLoss: {display_loss/it_display}"
                    )
                    
                    writer.add_scalar(
                            "training generator loss",
                            display_loss / it_display,
                            epoch * len(dataloader) + it,
                        )
                                        
                    display_loss = 0 
                    it_display = 0 
                    
                    with torch.no_grad():
                        real_audio = signal_in[:4].detach().cpu()
                        rec_audio = signal_out[:4].detach().cpu()
                        rec_harmonics = harmonics.detach().cpu().numpy()

                        pitchp = pitch[:4].detach().cpu().numpy()
                        loudp  = loud[:4].detach().cpu().numpy() 

                        for j in range(real_audio.shape[0]):
                            figure = plot_metrics(pitchp[j],loudp[j],real_audio[j].numpy(),rec_audio[j].numpy(),rec_harmonics[j],self.sr,self.frame_size)
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
                        
            if epoch % 10 == 0 or epoch == n_epochs - 1 :                
                torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.decoder.state_dict(),
                    "optimizer_state_dict": self._opt.state_dict(),
                    "loss": losses,
                }, os.path.join(logdir, "DDSP_lr_{lr}_n_epochs_{n_epochs}__sr_{self.sr}__frame_{self.frame_size}.pt"))