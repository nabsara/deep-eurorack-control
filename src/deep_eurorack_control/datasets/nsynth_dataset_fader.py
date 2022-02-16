import os
import json
import scipy.io.wavfile
from torch.utils.data import Dataset
import numpy as np


class NSynthDataset_Fader(Dataset):

    def __init__(self, data_dir, audio_dir, nsynth_json="nsynth_string.json", transform=None):
        super(NSynthDataset_Fader, self).__init__()
        with open(os.path.join(data_dir, nsynth_json), "r") as f:
            self.audio_labels = json.load(f)
        self.audio_name_list = list(self.audio_labels.keys())
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        audio_file = os.path.join(self.audio_dir, f"{self.audio_name_list[index]}.wav")
        sr, audio = scipy.io.wavfile.read(audio_file)
        audio = audio.astype(np.float32)
        if self.transform:
            audio = self.transform(audio)
        pitch = self.audio_labels[self.audio_name_list[index]]['pitch']
        random_param = np.random.uniform(0,1)
        return audio, pitch, random_param

