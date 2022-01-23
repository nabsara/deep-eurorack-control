from deep_eurorack_control.models.ddsp.model import DDSP
from deep_eurorack_control.datasets.ddsp import NSynth_ddsp
from torch.utils.data import DataLoader



class DDSP_Pipeline:
    
    def __init__(
        self,
        dataset_dir,
        sr,
        frame_size,
        n_harmonics,
        n_bands
        
    ):
        self.model = DDSP(sr,frame_size,n_harmonics,n_bands)
        
        self.dataset = NSynth_ddsp(dataset_dir)

        
    def train(self,lr,batch_size,n_epochs,display_step,logdir):
        dataloader = DataLoader(self.dataset,batch_size=batch_size, shuffle=True)
        self.model.train(dataloader,lr,n_epochs,display_step,logdir)
        