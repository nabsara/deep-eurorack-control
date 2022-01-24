import torch.nn as nn


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
    if isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
