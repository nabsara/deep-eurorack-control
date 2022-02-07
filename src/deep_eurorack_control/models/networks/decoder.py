import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
from deep_eurorack_control.models.networks.utils import mod_sigmoid, amp_to_impulse_response, fft_convolve


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(negative_slope=0.2),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=3, dilation=1, padding=dilation+1))
        )

        self.shortcut = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ResidualStack(nn.Module):

    def __init__(self, dim):
        super(ResidualStack, self).__init__()

        dilation = [1, 3, 9]
        blocks = [ResnetBlock(dim, dilation[i]) for i in range(len(dilation))]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class UpSamplingLayer(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, out_pad=0):
        super(UpSamplingLayer, self).__init__()

        self.net = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            weight_norm(nn.ConvTranspose1d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, output_padding=out_pad))
        )

    def forward(self, x):
        return self.net(x)


class NoiseSynthesizer(nn.Module):

    def __init__(self, in_dim, out_dim, ratios, noise_bands):
        super(NoiseSynthesizer, self).__init__()
        self.data_size = out_dim
        self.target_size = torch.tensor(np.prod(ratios)).long(),
        net = []

        for i in range(len(ratios) - 1):
            net.append(nn.Sequential(
                nn.Conv1d(in_dim, in_dim, 3, stride=ratios[i], padding=1),
                nn.LeakyReLU(negative_slope=0.2)
            ))

        # last layer
        net.append(nn.Sequential(
            nn.Conv1d(in_dim, out_dim * noise_bands, 3, stride=ratios[-1], padding=1),
            nn.LeakyReLU(negative_slope=0.2)  # cf. schema
        ))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        # TODO: add white noise + filter
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class Decoder(nn.Module):

    def __init__(self, data_size, latent_dim=128, hidden_dim=64, noise_ratios=[4, 4, 4], noise_bands=5):
        super(Decoder, self).__init__()

        ratios = [4, 4, 4, 2]

        # 1st conv layer
        net = [nn.Sequential(
            weight_norm(nn.Conv1d(latent_dim, 2**len(ratios) * hidden_dim, 7, stride=1, padding=3))
        )]

        # 4x : upsampling (LeakyReLU and convTransposed)
        # + residual stack (LeakyReLU and dilated conv)
        for i, r in enumerate(ratios):
            net.append(nn.Sequential(
                UpSamplingLayer(
                    in_dim=2**(len(ratios) - i) * hidden_dim,
                    out_dim=2**(len(ratios) - (i + 1)) * hidden_dim,
                    kernel_size=2 * r + 1,
                    stride=r,
                    padding=r - 1,
                    out_pad=1
                ),
                ResidualStack(dim=2**(len(ratios) - (i + 1)) * hidden_dim)
            ))

        # modified version of the generator proposed by Kumar
        # et al. (2019) ie. same alternation of upsampling
        # layers and residual networks but instead of directly
        # outputting the raw waveform we feed the last hidden
        # layer to three sub-networks.
        #self.net = nn.Sequential(*net)

        self.conv1 = weight_norm(nn.Conv1d(latent_dim, 2**len(ratios) * hidden_dim, 7, stride=1, padding=3))
        self.up1 = UpSamplingLayer(
                    in_dim=2**(len(ratios) - 0) * hidden_dim,
                    out_dim=2**(len(ratios) - (0 + 1)) * hidden_dim,
                    kernel_size=2 * 4 + 1,
                    stride=4,
                    padding=3,
                    out_pad=1,
                )
        self.res1 = ResidualStack(dim=2**(len(ratios) - (0 + 1)) * hidden_dim)
        self.up2 = UpSamplingLayer(
                    in_dim=2**(len(ratios) - 1) * hidden_dim,
                    out_dim=2**(len(ratios) - (1 + 1)) * hidden_dim,
                    kernel_size=2 * 4 + 1,
                    stride=4,
                    padding=3,
                    out_pad=1,
                )
        self.res2 = ResidualStack(dim=2**(len(ratios) - (1 + 1)) * hidden_dim)
        self.up3 = UpSamplingLayer(
                    in_dim=2**(len(ratios) - 2) * hidden_dim,
                    out_dim=2**(len(ratios) - (2 + 1)) * hidden_dim,
                    kernel_size=2 * 4 + 1,
                    stride=4,
                    padding=3,
                    out_pad=1,
                )
        self.res3 = ResidualStack(dim=2**(len(ratios) - (2 + 1)) * hidden_dim)
        self.up4 = UpSamplingLayer(
                    in_dim=2**(len(ratios) - 3) * hidden_dim,
                    out_dim=2**(len(ratios) - (3 + 1)) * hidden_dim,
                    kernel_size=2 * 2 + 1,
                    stride=2,
                    padding=2,
                    out_pad=1
                )
        self.res4 = ResidualStack(dim=2**(len(ratios) - (3 + 1)) * hidden_dim)


        # 1st subnetwork (waveform) synthesizes a multiband
        # audio signal (with tanh activation)
        self.waveform = nn.Sequential(
            weight_norm(nn.Conv1d(hidden_dim, data_size, 7, padding=3)),
            # nn.Tanh()
        )
        # 2nd sub-network (loudness), generating an amplitude
        # envelope (with sigmoid activation)
        self.loudness = nn.Sequential(
            weight_norm(nn.Conv1d(hidden_dim, 1, 3, stride=1, padding=1)),
            # nn.Sigmoid()
        )
        # 3rd sub-network noise synthesizer (proposed in
        # Engel et al. (2019)), and produces a multiband
        # filtered noise added to the previous signal.
        self.noise_synth = NoiseSynthesizer(
             in_dim=hidden_dim,
             out_dim=data_size,
             ratios=noise_ratios,
             noise_bands=noise_bands
        )

    def forward(self, x):
        # x_dec = self.net(x)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        x = self.up3(x)
        x = self.res3(x)
        x = self.up4(x)
        x_dec = self.res4(x)

        waveform = self.waveform(x_dec)
        loudness = self.loudness(x_dec)
        # output = waveform * loudness

        # TODO: implement NoiseSynthesizer forward
        noise = self.noise_synth(x_dec)
        output = torch.tanh(waveform) * mod_sigmoid(loudness) + noise
        return output
