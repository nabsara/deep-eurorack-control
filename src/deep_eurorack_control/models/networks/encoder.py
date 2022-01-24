import torch.nn as nn


class Encoder(nn.Module):
    """
    data_sise = 16   # equal to n_band in PQMF
    """
    def __init__(self, data_size, latent_dim=128, hidden_dim=64):
        super(Encoder, self).__init__()

        ratios = [4, 4, 4, 2]

        # 1st conv layer
        net = [nn.Sequential(
            nn.Conv1d(data_size, hidden_dim, 7, stride=1)
        )]

        # 4x conv batchnorm LeakyReLu
        for i, r in enumerate(ratios):
            net.append(nn.Sequential(
                nn.Conv1d(2**i * hidden_dim, 2**(i + 1) * hidden_dim, kernel_size=2 * r + 1, stride=r),
                nn.BatchNorm1d(2**(i + 1) * hidden_dim),
                nn.LeakyReLU(negative_slope=0.2)
            ))

        self.net = nn.Sequential(*net)

        self.mean = nn.Sequential(
            nn.Conv1d(2**len(ratios) * hidden_dim, latent_dim, 5)
        )

        self.var = nn.Sequential(
            nn.Conv1d(2**len(ratios) * hidden_dim, latent_dim, 5),
            nn.Softplus()
        )

    def forward(self, x):
        x_enc = self.net(x)
        mean = self.mean(x_enc)
        var = self.var(x_enc)
        return mean, var
