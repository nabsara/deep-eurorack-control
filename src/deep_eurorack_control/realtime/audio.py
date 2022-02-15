import time
import ctypes
import threading

import numpy as np
import sounddevice as sd


class Audio(threading.Thread):
    def __init__(self, q, sr, audio_block_size):
        self.sr = sr
        self.audio_block_size = audio_block_size
        print("[ ] - Audio block size:", audio_block_size)
        # sd.default.device = 0
        sd.default.samplerate = 16000.0
        print(sd.query_devices())
        print(sd.query_devices(0))
        threading.Thread.__init__(self, daemon=True)
        self.q = q
        # Current block stream
        self._cur_stream = None

    def __del__(self):
        self._cur_stream.close()

    def run(self):
        # For testing
        # audio = np.random.randn(5 * self.sr)
        # sd.play(audio, 48000.0)
        def callback_block(outdata, frames, time, status):
            cur_data = self.q.get()
            # print("curdata shape", cur_data.shape)
            if (cur_data is None):
                print('Stream stopping (end of features)')
                raise sd.CallbackStop()
            outdata[:] = cur_data[:, np.newaxis]

        if (self._cur_stream == None):
            self._cur_stream = sd.OutputStream(
                callback=callback_block,
                blocksize=self.audio_block_size,
                channels=1
            )
            print('[+] - Audio stream launched')
            self._cur_stream.start()
            
        elif (not self._cur_stream.active):
            self._cur_stream.close()
            self._cur_stream = sd.OutputStream(
                callback=callback_block,
                blocksize=self.audio_block_size,
                channels=1
            )
            print('[ ] - Audio stream restarted')
            self._cur_stream.start()
