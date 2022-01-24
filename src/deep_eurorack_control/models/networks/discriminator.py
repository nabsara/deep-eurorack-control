import torch.nn as nn
from torch.nn.utils import weight_norm


def initialize_weights(m):
    """
    Initialize the model weights to the normal distribution
    with mean 0 and standard deviation 0.02

    Parameters
    ----------
    m : nn.Module
        is instance of nn.Conv2d or nn.ConvTranspose2d or nn.BatchNorm2d
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class NLayerDiscriminator(nn.Module):

    def __init__(self):
        super(NLayerDiscriminator, self).__init__()
        net = nn.ModuleList()

        # 1st conv layer 1 --> 16
        net.append(nn.Sequential(
            weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
            nn.LeakyReLU(negative_slope=0.2)
        ))

        # 2-3-4-5 down sampling layer 16 --> 64 --> 256 --> 1024 --> 1024
        net.append(nn.Sequential(
            weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4)),
            nn.LeakyReLU(negative_slope=0.2),
            weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16)),
            nn.LeakyReLU(negative_slope=0.2),
            weight_norm(nn.Conv1d(256, 1024,  kernel_size=41, stride=4, groups=64)),
            nn.LeakyReLU(negative_slope=0.2),
            weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=256)),
            nn.LeakyReLU(negative_slope=0.2)
        ))

        # 6th conv layer 1024 --> 1024
        net.append(nn.Sequential(
            weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1)),
            nn.LeakyReLU(negative_slope=0.2)
        ))

        # 7th conv layer 1024 --> 1
        net.append(nn.Sequential(
            weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1))
        ))

        self.net = net

    def forward(self, x):
        outputs = []  # store feature map output of each block layer
        for layer in self.net:
            x = layer(x)
            outputs.append(x)
        return outputs


class Discriminator(nn.Module):
    """
    the exact same discriminator as in Kumar et al. (2019),
    which is a strided convolutional network applied on different
    scales of the audio signal to prevent artifacts.
    also use the same feature matching loss as in the original paper.

    multi-scale architecture with 3 discriminators (D1, D2, D3) that
    have identical network structure but operate on different audio
    scales. D1 operates on the scale of raw audio, whereas D2, D3
    operate on raw audio downsampled by a factor of 2 and 4 respectively.
    The downsampling is performed using strided average pooling with
    kernel size 4. Multiple discriminators at different scales are
    motivated from the fact that audio has structure at different levels.
    This structure has an inductive bias that each discriminator learns
    features for different frequency range of the audio. For example,
    the discriminator operating on downsampled audio, does not have
    access to high frequency component, hence, it is biased to learn
    discriminative features based on low frequency components only.

    """
    def __init__(self):
        super().__init__()

        self.net = nn.ModuleList()
        self.n_disc = 3
        for i in range(self.n_disc):
            self.net.append(NLayerDiscriminator())

        self.downsample = nn.AvgPool1d(4, stride=2)

        self.net.apply(initialize_weights)

    def forward(self, x):
        results = []
        for disc in self.net:
            results.append(disc(x))
            x = self.downsample(x)
        return results
