import time
import threading

import torch
import numpy as np

import tediumControl


class CV(threading.Thread):
    def __init__(self, q, start_idx, end_idx, default_enveloppes_folder):
        threading.Thread.__init__(self, daemon=True)
        self.q = q

        # Used to restric the range of the default envelopes (to remove silence)
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.read_time_delta = 0.01
        self.spi_opened = 0
        self.spifd = -1

        self.cv_values = []
        self.cv_values_buffer_size = self.end_idx - self.start_idx
        print(f"{self.cv_values_buffer_size} CV values buffer target size")
        # Active_enveloppes keeps track of which CVs are used
        # replace all non-used ones with the default_enveloppes
        self.active_CVs = [False, False, False, False, False, False]
        self.is_playing = False

        # self.default_enveloppes = self.load_default_enveloppes(default_enveloppes_folder)
        print("[+] - Default enveloppes loaded")

        ########################
        ## Tedium configuration
        ## Cv values: 1=pitch, 2=loudness, 3=bandwidth, 4=flatness
        ## The drum types will be handled with the buttons (TODO)
        # Init the ADC
        self.spifd = tediumControl.adc_open()
        if self.spifd < 0:
            print("[-] - Could not Connect to the ADC")
            exit(-1)  # DEBUG
        print("[+] - Connected to the ADC, spifd =", self.spifd)
        self.spi_opened = 1
        # Init the buffer to the max size (easier for audio to have a fixed length buffer)
        for v in range(6):
            self.cv_values.append(np.zeros(v))
        print ("[+] - CV read thread launched")

    def detect_active_cvs(self):
        """
        Goal is to detect if a certain CV is plugged in tedium:
        - Compute the variance on the last 4 values
        - If it goes over the treshold, tag the CV as active.
        """
        # Compute variance first, we'll delete the old values after
        for cv_idx, v in enumerate(self.cv_values):
            # Compute the variance on the last 4 values only
            var = np.var(v[-4:])
            if var > 1:
                self.active_CVs[cv_idx] = True
            else:
                self.active_CVs[cv_idx] = False

    def update_CV_values_buffer(self, new_cv_values):
        """
        - Store all new CV values in a dual array[cv_idx][time (most recent one at -1)]
        - Remove oldest values when buffer is full
        """
        for cv_idx, v in enumerate(new_cv_values):
            self.cv_values[cv_idx] = np.append(self.cv_values[cv_idx], v)
            if self.cv_values[cv_idx].size > self.cv_values_buffer_size:
                self.cv_values[cv_idx] = self.cv_values[cv_idx][1:]

    def get_final_pred_buffer(self):
        """
        Create the final_pred_buffer:
        - Use stored values for connected CVs
        - Replace by default enveloppes for inactive CVs
        """
        res = {}
        for idx, key in enumerate(["pitch", "loudness"]):
            # if not self.active_CVs[idx]:
            #     res[key] = self.default_enveloppes[key][:self.cv_values_buffer_size].unsqueeze(0)
            # else:
                print(f"[ ] - Using CV values for: {key}")
                res[key] = torch.from_numpy(
                    np.expand_dims(
                        np.expand_dims(self.cv_values[idx], -1),
                        0
                    ).astype(np.float)
                ).float()
                print(res[key])
        return res

    def run(self):
        """ Main timing function """
        while 42:
            start_time = time.time()
            self.do_turn()
            print(self.cv_values)
            sleep_time = self.read_time_delta - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[-] - CV thread Lagging {sleep_time}s behind")

    def do_turn(self):
        # adc_bang() Returns a tuple of 8 values
        # (only 6 input CVs tho, don't know what are the last two ones)
        new_cv_values = tediumControl.adc_bang(self.spifd)  # DEBUG
        # new_cv_values = [0, 0, 0, 0, 0, 4000]  # DEBUG
        self.update_CV_values_buffer(new_cv_values)
        # self.detect_active_cvs()
        # Play only if the 6th CV value is > 2000
        if new_cv_values[2] > 2000:
            if not self.is_playing:
                self.is_playing = True
                # If there is still something in the queue it's outdated, remove everything
                with self.q.mutex:
                    self.q.queue.clear()
                self.q.put(self.get_final_pred_buffer())
            else:
                pass
                # We do not want to play if we've already launched a prediction
                # We'll play at the next NEW prediction trigger
        else:
            if self.is_playing:
                self.is_playing = False

    # def load_default_enveloppes(self, default_enveloppes_folder):
    #     pitch = torch.from_numpy(np.load(f"./{default_enveloppes_folder}/pitch.npy"))
    #     pitch = pitch[0][self.start_idx:self.end_idx]
    #     loudness = torch.from_numpy(np.load(f"./{default_enveloppes_folder}/loudness.npy"))
    #     loudness = loudness[0][self.start_idx:self.end_idx]
    #     descriptors = torch.from_numpy(np.load(f"./{default_enveloppes_folder}/descriptors.npy"))
    #     descriptors = descriptors[0][self.start_idx:self.end_idx]
    #     return {"pitch": pitch, "loudness": loudness, "descriptors": descriptors}
