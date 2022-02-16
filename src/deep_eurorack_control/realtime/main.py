from cv import CV
import queue
from audio import Audio
import torch
import math
import samplerate
import time
import os
from deep_eurorack_control.realtime.utils import load_model,generate_signal_realtime

import numpy as np


class DdspTedium(object):

    def __init__(self):
        super(DdspTedium, self).__init__()
        torch.set_grad_enabled(False)
        # self.run_name = run_name

        # torch.set_grad_enabled(False)

        # self.config = load_config(self.run_name + "/config.yaml")
        # print("[+] - Config loaded")
        # self.model = load_model(self.config, self.run_name)
        # print("[+] - Model loaded")

        # self.start_idx = 0
        # self.end_idx = 70
        # audio_block_size = self.end_idx - self.start_idx
        # audio_block_size *= self.config["preprocess"]["block_size"]  # Model upscaling
        # audio_block_size *= 3  # Audio upscaling
        audio_block_size = int(  256*48000*(1.0 / 16000)   )
        self.sr = 48000
        # signal.signal(signal.SIGINT, self.signal_handler)
        self.cv_q = queue.Queue()
        self.cv_thread = CV(self.cv_q)
        self.cv_thread.daemon = True
        self.audio_q = queue.Queue(10)
        self.audio_thread = Audio(self.audio_q, self.sr, audio_block_size)
        self.audio_thread.daemon = True

        n_bands = 65; n_harmonics = 51; frame_size = 256; sr = 16000; 
        logdir = "/home/pi/Desktop/atiam/deep-eurorack-control/data"
        checkpoint_path = "DDSP_lr_{lr}_n_epochs_{n_epochs}__sr_{self.sr}__frame_{self.frame_size}.pt"
        checkpoint_path = os.path.join(logdir,checkpoint_path)
        self.model = load_model(checkpoint_path,sr,frame_size,n_harmonics,n_bands)


    
    def launch(self):
        self.cv_thread.start()
        self.audio_thread.start()
        i=0

        time_list= []
        init_phase = torch.tensor([0])
        while i<100:
            start = time.time()
            if self.cv_q.empty():
                pass
            # print('hi')
            final_pred_buffer = self.cv_q.get()


            # for k, v in final_pred_buffer.items():
            #     print(f"{k}: {v.shape} / {v.type()}")
            pitch = final_pred_buffer["pitch"]
            # loudness = final_pred_buffer["loudness"]
            loudness = torch.ones_like(pitch)
            # print(pitch)
            # harmonics,filters = self.model.decoder.real_time_forward(pitch,loudness)
            # audio,final_phase = generate_signal_realtime(pitch,harmonics,filters,self.model.frame_size,self.model.sr,init_phase)
            # init_phase = final_phase
            #
            audio = torch.rand(1,256)
            # audio = np.random.randn(768)
            audio = samplerate.resample(audio[0], int(48000 * (1.0 / 16000)), 'sinc_best')
            
            print(audio.shape)
            
            self.audio_q.put(audio)
      
      
            end = time.time()
            time_list.append(end-start)

            i+=1

        print(torch.mean(torch.tensor(time_list)))
        # self.audio_thread.__del__()

if __name__ == '__main__':
    app = DdspTedium()
    app.launch()
