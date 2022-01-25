import os
import torch
import torch.distributions as distributions
from tqdm import tqdm
import time

from deep_eurorack_control.config import settings
from deep_eurorack_control.models.networks import Encoder, Decoder, Discriminator


class RAVE:

    def __init__(self, n_band=16, latent_dim=128, hidden_dim=64, sampling_rate=48000):

        self.multi_band_decomposition = None  # PQMF

        data_size = n_band
        self.encoder = Encoder(
            data_size=data_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        self.decoder = Decoder(
            data_size=data_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        self.discriminator = Discriminator()

        self.adv_train = False

    def _init_optimizer(self):
        pass

    def _init_criterion(self):
        pass

    @staticmethod
    def sampling(x, mu, sigma):
        n_batch = x.shape[0]
        q = distributions.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        epsilon = q.sample((int(n_batch),))
        z = mu + sigma * epsilon
        return z

    @staticmethod
    def kl_div_loss(x, mu, sigma):
        n_batch = x.shape[0]
        kl_div = 0.5 * torch.sum(1 + sigma - torch.pow(mu, 2) - torch.exp(sigma))
        return kl_div / n_batch

    def train_step(self, data_current_batch):
        """Inside a Batch"""
        # STEP 1:
        # multi band decomposition pqmf

        # Encode data
        # if train step 1 repr learning encoder.train()
        # else (train step 2 adversarial) freeze encoder encoder.eval()

        # get latent space samples

        # Decode latent space samples

        # compute regularization loss
        # compute reconstruction loss

        # STEP 2:
        # inverse multi band decomposition (pqmf -1)

        # compute discriminator loss on fake and real data

        # FINALLY:
        # compute decoder-generator loss

        # optimizer steps:
        # Before the backward pass, zero all of the network gradients
        # self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        # loss.backward()
        # Calling the step function to update the parameters
        # self.optimizer.step()
        pass

    def validation_step(self, current_batch):
        # For VAE training step 1 repr learning
        # model.eval() mode

        # 1. multi band decomposition pqmf

        # 2. Encode data

        # 3. get latent space samples

        # 4. Decode latent space samples

        # 5. compute total (reconstruction + regularization loss)
        pass

    def train(self, train_loader, valid_loader, lr, n_epochs, display_step, models_dir, model_filename):
        start = time.time()

        self._init_optimizer()
        self._init_criterion()

        train_losses = []
        valid_losses = []
        it = 0  # number of batch iterations updated at the end of the DataLoader for loop
        for epoch in range(n_epochs):
            cur_step = 0
            it_display = 0
            loss_display = 0
            for x, _ in tqdm(train_loader):
                x = x.to(settings.device)
                self.train_step(x)

                loss = 1  # TODO: retrieve losses from train_step()

                # keep track of the loss
                loss_display += loss.item()
                it_display += 1
                if it % display_step == 0 or (
                        (epoch == n_epochs - 1) and (cur_step == len(train_loader) - 1)
                ):
                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{cur_step}/{len(train_loader)}]"
                        f"\tTime: {time.time() - start} (s) \tTotal_loss: {loss_display / it_display}"
                    )
                    train_losses.append(loss_display / it_display)
                    loss_display = 0
                    it_display = 0
                cur_step += 1
                it += 1

            with torch.no_grad():
                for x, _ in tqdm(valid_loader):
                    x = x.to(settings.device)
                    self.validation_step(x)

                    loss = 1  # TODO: retrieve losses from validation_step()
                    loss_display += loss.item()
                    it_display += 1
                print(
                    f"\nEpoch: [{epoch}/{n_epochs}] \t Validation loss: {loss_display / it_display}"
                )
                valid_losses.append(loss_display / it_display)

            # model checkpoints:
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": train_losses[-1]
                    },
                    os.path.join(models_dir, model_filename)
                )
