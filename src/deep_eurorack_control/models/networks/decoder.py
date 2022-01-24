import torch.nn as nn


class ResidualStack(nn.Module):

    def __init__(self):
        super(ResidualStack, self).__init__()

    def forward(self, x):
        pass


class NoiseSynthesizer(nn.Module):

    def __init__(self):
        super(NoiseSynthesizer, self).__init__()



class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        blocks = []

        block = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose1d(),
            ResidualStack()
        )

        # modified version of the generator proposed by Kumar
        # et al. (2019) ie. same alternation of upsampling
        # layers and residual networks but instead of directly
        # outputting the raw waveform we feed the last hidden
        # layer to three sub-networks.
        self.net = nn.Sequential(*blocks)
        # 1st subnetwork (waveform) synthesizes a multiband
        # audio signal (with tanh activation)
        self.waveform = nn.Sequential()
        # 2nd sub-network (loudness), generating an amplitude
        # envelope (with sigmoid activation)
        self.loudness = nn.Sequential()
        # 3rd sub-network noise synthesizer (proposed in
        # Engel et al. (2019)), and produces a multiband
        # filtered noise added to the previous signal.
        self.noise_synth = NoiseSynthesizer()

    def forward(self, x):
        x_dec = self.net(x)
        waveform = self.waveform(x_dec)
        loudness = self.loudness(x_dec)
        noise = self.noise_synth(x_dec)
        output = waveform * loudness + noise
        return output
