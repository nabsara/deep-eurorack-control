from deep_eurorack_control.datasets.data_loaders import nsynth_data_loader
from deep_eurorack_control.models.rave import RAVE


class RAVEPipeline:

    def __init__(self, data_dir, audio_dir, nsynth_json, models_dir, batch_size, n_band=16, latent_dim=128, hidden_dim=64, sampling_rate=16000, use_noise=False, init_weights=False):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.models_dir = models_dir
        self.batch_size = batch_size
        self.nsynth_json = nsynth_json

        self.train_loader, self.valid_loader = nsynth_data_loader(
            batch_size=self.batch_size,
            data_dir=self.data_dir,
            audio_dir=self.audio_dir,
            nsynth_json=self.nsynth_json
        )

        self.model = RAVE(
            n_band=n_band,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            sampling_rate=sampling_rate,
            use_noise=use_noise,
            init_weights=init_weights
        )

    def train(self, learning_rate, n_epochs, display_step, n_epoch_warmup, seed):
        model_filename = f"{self.model.model_name}__b_{self.batch_size}__lr_{learning_rate}__e_{n_epochs}__e_warmup_{n_epoch_warmup}__seed_{seed}"

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
