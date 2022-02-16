from deep_eurorack_control.datasets.data_loaders import mnist_data_loader
from deep_eurorack_control.models.vae import ModelVAE


class MNISTPipeline:

    def __init__(self, data_dir, models_dir, batch_size):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.batch_size = batch_size

        self.train_loader, self.valid_loader, self.test_loader = mnist_data_loader(
            self.batch_size, self.data_dir
        )

        self.model = ModelVAE()

    def train(self, learning_rate, n_epochs, display_step, show_fig=False):
        model_filename = f"{self.model.model_name}__b_{self.batch_size}__lr_{learning_rate}__e_{n_epochs}.pt"

        self.model.train(
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            lr=learning_rate,
            n_epochs=n_epochs,
            display_step=display_step,
            models_dir=self.models_dir,
            model_filename=model_filename,
            show_fig=show_fig
        )