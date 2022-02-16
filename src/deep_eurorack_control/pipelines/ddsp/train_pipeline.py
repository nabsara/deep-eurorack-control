from deep_eurorack_control.models.ddsp.model import DDSP
from deep_eurorack_control.datasets.ddsp import NSynth_ddsp
from torch.utils.data import DataLoader



class DDSP_Pipeline:
    
    def __init__(
        self,
        model_name,
        dataset_dir,
        sr,
        frame_size,
        n_harmonics,
        n_bands,
        n_hidden,
        residual,
        n_z
        
    ):
        self.model = DDSP(model_name,sr,frame_size,n_harmonics,n_bands,n_hidden,residual,n_z)
        
        self.dataset = NSynth_ddsp(dataset_dir)

        
    def train(self,lr,batch_size,n_epochs,display_step,logdir,alpha):
        dataloader = DataLoader(self.dataset,batch_size=batch_size, shuffle=True)
        self.model.train(dataloader,lr,n_epochs,display_step,logdir,alpha)
        