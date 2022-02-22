import torch
import torch.nn as nn
import os
import click
from tqdm import tqdm

import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from deep_eurorack_control.models.rave import RAVE
from deep_eurorack_control.datasets.data_loaders import nsynth_data_loader
from deep_eurorack_control.config import settings


mel_basis = {}
hann_window = {}
device = settings.device


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(x, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(x) < -1.:
    #    print('min value is ', torch.min(x))
    # if torch.max(x) > 1.:
    #    print('max value is ', torch.max(x))

    global mel_basis, hann_window, device
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax) + '_' + str(x.device)] = torch.from_numpy(mel).float().to(x.device)
        hann_window[str(x.device)] = torch.hann_window(win_size).to(x.device)

    x = torch.nn.functional.pad(x.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    x = x.squeeze(1)

    melspectro = torch.stft(x, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(x.device)],
                            center=center, pad_mode='reflect', normalized=False, onesided=True)

    melspectro = torch.sqrt(melspectro.pow(2).sum(-1) + (1e-9))

    melspectro = torch.matmul(mel_basis[str(fmax) + '_' + str(x.device)], melspectro)
    melspectro = spectral_normalize_torch(melspectro)

    return melspectro


def melspectrogram_loss(x, x_gen, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    x_mel = mel_spectrogram(x, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center).to(settings.device)
    x_gen_mel = mel_spectrogram(x_gen, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center).to(settings.device)

    loss_melspectro = F.l1_loss(x_mel, x_gen_mel)

    return loss_melspectro


def inference(model, test_loader):
    x_list = []
    y_list = []
    for x, _ in tqdm(test_loader):
        x_list.append(x)
        x = torch.reshape(x, (x.shape[0], 1, -1)).to(settings.device)

        # 1. multi band decomposition pqmf
        x = model.multi_band_decomposition(x)

        # 2. Encode data
        mean, var = model.encoder(x)

        # z, _ = model.reparametrize(mean, var)
        z = mean

        y = model.decoder(z)
        y = model.multi_band_decomposition.inverse(y)
        y = y.reshape(y.shape[0], -1)
        y_list.append(y)

    x_test = torch.cat(x_list, 0)
    y_test = torch.cat(y_list, 0)
    return x_test, y_test


@click.option(
    "--data_dir",
    default=settings.DATA_DIR,
    help="Absolute path to data directory",
)
@click.option(
    "--audio_dir",
    default=settings.AUDIO_DIR,
    help="Absolute path to audio .wav directory",
)
@click.option(
    "--models_dir",
    default=settings.MODELS_DIR,
    help="Absolute path to models directory",
)
@click.option(
    "--checkpoint_file",
    default="n_synth_rave__n_band_8__latent_128__sr_16000__noise_True__init_weights_True__b_8__lr_0.0001__e_250__e_warmup_150__vae.pt",
    help="model checkpoint",
)
@click.option(
    "--nsynth_json",
    default="nsynth_string_test.json",
    help="Nsynth JSON audio files selection"
)
@click.option(
    "--n_band",
    default=8,
    help="Number of bands in the multiband signal decomposition (pqmf)",
)
@click.option("--noise", is_flag=True)
def evaluate(data_dir, audio_dir, models_dir, checkpoint_file, nsynth_json, n_band, noise):
    n_fft = 1024
    num_mels = 80
    sampling_rate = 16000
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 8000

    test_loader, _ = nsynth_data_loader(
        batch_size=n_band,
        data_dir=data_dir,
        audio_dir=audio_dir,
        nsynth_json=nsynth_json,
        valid_ratio=0.
    )

    checkpoint = torch.load(os.path.join(models_dir, checkpoint_file), map_location=settings.device)

    model = RAVE(
        n_band=n_band,
        latent_dim=128,
        hidden_dim=64,
        sampling_rate=16000,
        use_noise=noise,
        init_weights=True
    )

    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.encoder.eval()
    model.decoder.eval()

    # x, y = inference(model, test_loader)
    losses = []
    for s, _ in tqdm(test_loader):
        x = torch.reshape(s, (s.shape[0], 1, -1)).to(settings.device)

        # 1. multi band decomposition pqmf
        x = model.multi_band_decomposition(x)

        # 2. Encode data
        mean, var = model.encoder(x)

        # z, _ = model.reparametrize(mean, var)
        z = mean

        y = model.decoder(z)
        y = model.multi_band_decomposition.inverse(y)
        y = y.reshape(y.shape[0], -1)
        loss = melspectrogram_loss(s, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
        losses.append(loss)
    print(len(losses))
    mel_loss = torch.mean(losses)
    print(mel_loss)


@click.group()
def main():
    pass


main.command()(evaluate)


if __name__ == "__main__":
    main()
