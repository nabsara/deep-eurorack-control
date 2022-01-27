from deep_eurorack_control.datasets.data_loaders import nsynth_data_loader
from deep_eurorack_control.models.rave import RAVE


class RAVEPipeline:

    def __init__(self, data_dir, audio_dir, models_dir, batch_size):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.models_dir = models_dir
        self.batch_size = batch_size

        self.train_loader, self.valid_loader = nsynth_data_loader(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            audio_dir=self.audio_dir,
        )

        self.model = RAVE(n_band=1, latent_dim=128, hidden_dim=64, n_taps=4, sampling_rate=16000)

    def train(self, learning_rate, n_epochs, display_step, n_epoch_warmup):
        model_filename = f"{self.model.model_name}__b_{self.batch_size}__lr_{learning_rate}__e_{n_epochs}"

        self.model.train(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            lr=learning_rate,
            n_epochs=n_epochs,
            display_step=display_step,
            models_dir=self.models_dir,
            model_filename=model_filename,
            n_epoch_warmup=n_epoch_warmup
        )
