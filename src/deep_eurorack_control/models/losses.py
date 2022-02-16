import torch
import torch.nn as nn


class SpectralLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.scales = [2048, 1024, 512, 256, 128, 64]
        self.overlap = 0.75

    @staticmethod
    def lin_distance(x, y):
        return torch.norm(x - y) / torch.norm(x)

    @staticmethod
    def log_distance(x, y):
        return torch.mean(torch.abs(torch.log(x + 1e-7) - torch.log(y + 1.e-7)))  # abs because L1 norm

    def forward(self, x, y):
        spectral_dist_each_scale_list = []
        x_flat = x.reshape(-1, x.shape[2])  # to compute stft we need to flatten the band dim with the batch dim
        y_flat = y.reshape(-1, y.shape[2])
        for scale in self.scales:
            stft_x = torch.abs(torch.stft(
                input=x_flat,
                n_fft=scale,
                hop_length=int(scale * (1 - self.overlap)),
                win_length=scale,
                window=torch.hann_window(scale).to(x_flat),
                center=True,
                normalized=True,
                return_complex=True
            ))
            stft_y = torch.abs(torch.stft(
                input=y_flat,
                n_fft=scale,
                hop_length=int(scale * (1 - self.overlap)),
                win_length=scale,
                window=torch.hann_window(scale).to(y_flat),
                center=True,
                normalized=True,
                return_complex=True
            ))
            stft_x = stft_x.reshape(x.shape[0], x.shape[1], stft_x.shape[1], stft_x.shape[2]).reshape(x.shape[0], -1)
            stft_y = stft_y.reshape(y.shape[0], y.shape[1], stft_y.shape[1], stft_y.shape[2]).reshape(y.shape[0], -1)
            spectral_dist = self.lin_distance(stft_x, stft_y) + self.log_distance(stft_x, stft_y)
            spectral_dist_each_scale_list.append(spectral_dist)
        return sum(spectral_dist_each_scale_list)


class HingeLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.mean(torch.maximum(1 - torch.mul(output, target), 0 * target))


class LinearLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return torch.mean(torch.mul(target, output))


class FeatureMatchingLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, disc_features_real, disc_features_fake):
        fm = 0
        for feat_real_i, feat_fake_i in zip(disc_features_real, disc_features_fake):
            fm += torch.mean(torch.abs(feat_real_i - feat_fake_i))  # abs because L1 norm
        return fm / len(disc_features_real)
