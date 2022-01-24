import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, data_size, latent_dim=128):
        super(Encoder, self).__init__()
        self.data_size = data_size  # equal to n_band in PQMF
        self.latent_dim = latent_dim
        self.hidden_dims = [64, 128, 256, 512]
        self.strides = [4, 4, 4, 2]

        layers_dim = [self.data_size] + self.hidden_dims
        kernels = [7] + [2 * s + 1 for s in self.strides]
        self.strides.insert(0, 1)
        blocks = []
        for i in range(len(layers_dim) - 1):
            blocks.append(nn.Sequential(
                nn.Conv1d(layers_dim[i], layers_dim[i+1], kernels[i], stride=self.strides[i]),
                nn.BatchNorm1d(layers_dim[i+1]),
                nn.LeakyReLU(negative_slope=0.2)
            ))

        self.net = nn.Sequential(*blocks)

        self.mean = nn.Sequential(
            nn.Conv1d(self.hidden_dims[-1], self.latent_dim, 5)
        )

        self.var = nn.Sequential(
            nn.Conv1d(self.hidden_dims[-1], self.latent_dim, 5),
            nn.Softplus()
        )

    def forward(self, x):
        x_enc = self.net(x)
        mean = self.mean(x_enc)
        var = self.var(x_enc)
        return mean, var
