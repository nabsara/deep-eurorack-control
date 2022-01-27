import click

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
    "--batch_size",
    default=8,
    help="Data loader batch size",
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
def train_rave(
    data_dir,
    audio_dir,
    models_dir,
    batch_size,
    n_epochs,
    learning_rate,
    display_step,
    n_epoch_warmup,
):
    print(locals())
    pipeline = RAVEPipeline(
        data_dir=data_dir,
        audio_dir=audio_dir,
        models_dir=models_dir,
        batch_size=batch_size
    )
    pipeline.train(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        display_step=display_step,
        n_epoch_warmup=n_epoch_warmup
    )
