import click
import json
import os

from deep_eurorack_control.config import settings
from deep_eurorack_control.pipelines.mnist_pipeline import MNISTPipeline
from deep_eurorack_control.pipelines.rave_pipeline import RAVEPipeline


@click.option(
    "--data_dir",
    default=settings.DATA_DIR,
    help="Absolute path to data directory",
)
@click.option(
    "--models_dir",
    default=settings.MODELS_DIR,
    help="Absolute path to models directory",
)
@click.option(
    "--batch_size",
    default=128,
    help="Data loader batch size",
)
@click.option(
    "--n_epochs",
    default=50,
    help="Number of epochs",
)
@click.option(
    "--learning_rate",
    default=0.0001,
    help="Learning rate",
)
@click.option(
    "--display_step",
    default=500,
    help="Number of iterations between each training stats display",
)
@click.option("--show", is_flag=True)
def train_mnist_vae(
    data_dir,
    models_dir,
    batch_size,
    n_epochs,
    learning_rate,
    display_step,
    show,
):
    print(locals())
    pipeline = MNISTPipeline(data_dir, models_dir, batch_size)
    pipeline.train(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        display_step=display_step,
        show_fig=show
    )


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
    "--nsynth_json",
    default="nsynth_string.json",
    help="Nsynth JSON audio files selection"
)
@click.option(
    "--batch_size",
    default=8,
    help="Data loader batch size",
)
@click.option(
    "--n_band",
    default=16,
    help="Number of bands in the multiband signal decomposition (pqmf)",
)
@click.option(
    "--n_epochs",
    default=10,
    help="Number of epochs",
)
@click.option(
    "--learning_rate",
    default=0.0001,
    help="Learning rate",
)
@click.option(
    "--display_step",
    default=500,
    help="Number of iterations between each training stats display",
)
@click.option(
    "--n_epoch_warmup",
    default=2,
    help="Number of epoch for the first training stage representation learning"
)
@click.option(
    "--sampling_rate",
    default=16000,
    help="sampling rate",
)
@click.option("--noise", is_flag=True)
def train_rave(
    data_dir,
    audio_dir,
    models_dir,
    nsynth_json,
    batch_size,
    n_band,
    n_epochs,
    learning_rate,
    display_step,
    n_epoch_warmup,
    sampling_rate,
    noise
):
    print(locals())
    pipeline = RAVEPipeline(
        data_dir=data_dir,
        audio_dir=audio_dir,
        models_dir=models_dir,
        nsynth_json=nsynth_json,
        batch_size=batch_size,
        n_band=n_band,
        latent_dim=128,
        hidden_dim=64,
        sampling_rate=sampling_rate,
        use_noise=noise
    )
    pipeline.train(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        display_step=display_step,
        n_epoch_warmup=n_epoch_warmup
    )


@click.option(
    "--nsynth_path",
    default=settings.DATA_DIR,
    help="Absolute path to nsynth dataset directory",
)
@click.option(
    "--data_dir",
    default=settings.DATA_DIR,
    help="Absolute path to data directory",
)
@click.option(
    "--instrument_class",
    default="string",
    help="instrument class name",
)
@click.option(
    "--output_filename",
    default="nsynth_string.json",
    help="Absolute path to data directory",
)
def write_nsynth_json(nsynth_path, data_dir, instrument_class, output_filename):
    with open(os.path.join(nsynth_path, "examples.json"), "r") as f:
        data = json.load(f)

    data_strings = {k: v for k, v in data.items() if k.startswith(instrument_class)}
    print(f"nb samples : {len(data_strings.keys())}")

    with open(os.path.join(data_dir, output_filename), "w", encoding="utf-8") as f:
        json.dump(data_strings, f, indent=4)
