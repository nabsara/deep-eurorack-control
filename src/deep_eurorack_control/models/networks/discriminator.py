import torch.nn as nn


class Discriminator(nn.Module):
    """
    the exact same discriminator as in Kumar et al. (2019),
    which is a strided convolutional network applied on different
    scales of the audio signal to prevent artifacts.
    also use the same feature matching loss as in the original paper.

    """
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(Discriminator, self).__init__()

        layers_dim = [16, 64, 256, 1024]
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, )
        )

    def forward(self, x):
        pass
