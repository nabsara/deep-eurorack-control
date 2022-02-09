import os 

from deep_eurorack_control.helpers.utils import load_pickle
from torch.utils.data import Dataset
import torch
from scipy.signal import savgol_filter


class NSynth_ddsp(Dataset):
    """

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self, dataset_dir) -> None:
        super().__init__()
    
        self.p_arr = load_pickle(os.path.join(dataset_dir,'pitch.pkl'))
        self.pconf_arr = load_pickle(os.path.join(dataset_dir,'pitch_conf.pkl'))
        self.l_arr = load_pickle(os.path.join(dataset_dir,'loudness.pkl'))
        self.a_arr = load_pickle(os.path.join(dataset_dir,'audio.pkl'))

    def __len__(self):
        return self.p_arr.shape[0]

    def __getitem__(self, index):
        
        pitch = torch.tensor(self.p_arr[index].copy()).float()
        pitch_conf = torch.tensor(self.pconf_arr[index].copy()).float()
        loud = torch.tensor(self.l_arr[index].copy()).float()
        audio = torch.tensor(self.a_arr[index].copy()).float()
        return pitch,pitch_conf,loud,audio