import torch.nn as nn
import numpy as np


# prototype filter of length M through Kaiser window approach
def prototype_filter(len_filter, cutoff):
    """
    use impulse response of the ideal low-pass filter with cutoff frequency
    and then use window method to get finite impulse response filter
    """
    len_range = np.arange(len_filter)

    # choice of Kaiser window
    beta = 14  # parameter of the Kaiser window
    w = np.kaiser(len_filter, beta)

    v = np.divide(np.sin(cutoff * (len_range - 0.5 * len_filter)), np.pi * (len_range - 0.5 * len_filter))
    if len_filter % 2 == 0:
        v[int(len_filter / 2)] = cutoff / np.pi

    f = np.multiply(v, w)  # filter
    return f


# cosine modulation factors
def cos_modulation(n, k, n_band, m):
    phi_k = (-1) ** k + np.pi / 4
    result = np.cos((k + 0.5) * (n - (m - 1) / 2) * (np.pi / n_band) + phi_k)
    return result


def conv(signal_in, h):
    signal_out = np.convolve(signal_in, h, mode='same')
    return signal_out


class PQMF(nn.Module):
    """
    Pseudo Quadrature Mirror Filter multiband decomposition / reconstruction
    """
    def __init__(self, n_band=16, n_taps=8, sampling_rate=48000):
        super().__init__()
        self.n_band = n_band
        self.n_taps = n_taps
        self.len_filter = n_band * n_taps
        self.sampling_rate = sampling_rate

        self.cutoff_freq_Hz = sampling_rate / (2 * n_band)  # cutoff frequency for the prototype filter (Hz)
        self.cutoff_freq = np.pi / n_band  # cutoff frequency for the prototype filter (normalize by pi)

        self.h_analysis, self.h_synthesis = self._compute_analysis_and_synthesis_filers()

    def _compute_analysis_and_synthesis_filers(self):
        # analysis and synthesis filters
        h_analysis = np.zeros((self.n_band, self.len_filter))
        h_analysis[0] = prototype_filter(self.len_filter, self.cutoff_freq)
        h_synthesis = np.zeros((self.n_band, self.len_filter))
        h_synthesis[0] = h_analysis[0][::-1]  # same as analysis filter but reverse in time
        # since it is an orthogonal filter bank by construction the synthesis filters
        # are simply the time-reverse of the analysis filter

        scale = np.arange(self.len_filter)

        for k in np.arange(1, self.n_band):
            cos_mod = cos_modulation(scale, k, self.n_band, self.len_filter)
            h_analysis[k] = np.multiply(h_analysis[0], cos_mod)

            # defining the synthesis filters as time-reverse of the analysis filters
            h_synthesis[k] = h_analysis[k][::-1]

        return h_analysis, h_synthesis

    def forward(self, x):
        # pqmf analysis
        # (n_band, len_signal)
        decomposition = np.zeros((self.n_band, x.shape[-1]))

        decomposition = nn.functional.conv1d(
            x,
            self.h_analysis.unsqueeze(1),
            stride=self.h_analysis.shape[0],
            padding=self.h_analysis.shape[-1] // 2,
        )[..., :-1]
        # for k in np.arange(self.n_band):
            # decomposition[k] = conv(x, self.h_analysis[k])


        return decomposition

    def inverse(self, x):
        # pqmf synthesis
        reconstruction = None
        for k in np.arange(self.n_band):
            reconstruction += conv(x[k], self.h_synthesis[k])

        return reconstruction
