import torch.nn as nn

class EncoderAE(nn.Module):
    """
    data_size = 16   # equal to n_band in PQMF
    """
    def __init__(self, data_size, latent_dim=128, hidden_dim=64):
        super(EncoderAE, self).__init__()

        ratios = [4, 4, 4, 2]

        # 1st conv layer
        net = [nn.Sequential(
            nn.Conv1d(data_size, hidden_dim, 7, stride=1, padding=3)
        )]

        # 4x conv batchnorm LeakyReLu
        for i, r in enumerate(ratios):
            net.append(nn.Sequential(
                nn.Conv1d(2**i * hidden_dim, 2**(i + 1) * hidden_dim, kernel_size=2 * r + 1, stride=r, padding=r),
                nn.BatchNorm1d(2**(i + 1) * hidden_dim),
                nn.LeakyReLU(negative_slope=0.2)
            ))

        self.net = nn.Sequential(*net)

        # self.conv0 = nn.Conv1d(data_size, hidden_dim, 7, stride=1, padding=3)
        # self.conv1 = nn.Sequential(
        #         nn.Conv1d(2**0 * hidden_dim, 2**(0 + 1) * hidden_dim, kernel_size=2 * 4 + 1, stride=4, padding=4),
        #         nn.BatchNorm1d(2**(0 + 1) * hidden_dim),
        #         nn.LeakyReLU(negative_slope=0.2)
        #     )
        # self.conv2 = nn.Sequential(
        #         nn.Conv1d(2**1 * hidden_dim, 2**(1 + 1) * hidden_dim, kernel_size=2 * 4 + 1, stride=4, padding=4),
        #         nn.BatchNorm1d(2**(1 + 1) * hidden_dim),
        #         nn.LeakyReLU(negative_slope=0.2)
        #     )
        # self.conv3 = nn.Sequential(
        #         nn.Conv1d(2**2 * hidden_dim, 2**(2 + 1) * hidden_dim, kernel_size=2 * 4 + 1, stride=4, padding=4),
        #         nn.BatchNorm1d(2**(2 + 1) * hidden_dim),
        #         nn.LeakyReLU(negative_slope=0.2)
        #     )
        # self.conv4 = nn.Sequential(
        #         nn.Conv1d(2**3 * hidden_dim, 2**(3 + 1) * hidden_dim, kernel_size=2 * 2 + 1, stride=2, padding=2),
        #         nn.BatchNorm1d(2**(3 + 1) * hidden_dim),
        #         nn.LeakyReLU(negative_slope=0.2)
        #     )

    def forward(self, x):
        x_enc = self.net(x)
        return x_enc
