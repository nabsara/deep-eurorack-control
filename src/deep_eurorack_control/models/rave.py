import os
import torch
import torch.distributions as distributions
from tqdm import tqdm
import time

from deep_eurorack_control.config import settings
from deep_eurorack_control.models.networks import Encoder, Decoder, Discriminator
from deep_eurorack_control.models.networks.pqmf import PQMF
from deep_eurorack_control.models.losses import SpectralLoss, LinearLoss, HingeLoss, FeatureMatchingLoss


class RAVE:

    def __init__(self, n_band=16, latent_dim=128, hidden_dim=64, n_taps=8, sampling_rate=48000):

        self.model_name = "n_synth_rave"
        self.multi_band_decomposition = PQMF(
            n_band=n_band,
            n_taps=n_taps,
            sampling_rate=sampling_rate
        )

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

        self.warmed_up = False

    def _init_optimizer(self, learning_rate, beta_1=0.5, beta_2=0.9):
        param_vae = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.vae_opt = torch.optim.Adam(param_vae, lr=learning_rate, betas=(beta_1, beta_2))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

    def _init_criterion(self):
        self.spectral_dist_criterion = SpectralLoss()
        self.disc_criterion = HingeLoss()
        self.gen_criterion = LinearLoss()
        self.feat_matching_criterion = FeatureMatchingLoss()

    @staticmethod
    def sampling(x, mu, sigma):
        n_batch = x.shape[0]
        q = distributions.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        epsilon = q.sample((int(n_batch),))
        z = mu + sigma * epsilon.unsqueeze(2)
        return z

    @staticmethod
    def kl_div_loss(x, mu, sigma):
        n_batch = x.shape[0]
        kl_div = 0.5 * torch.sum(1 + sigma - torch.pow(mu, 2) - torch.exp(sigma))
        return kl_div / n_batch

    def train_step(self, data_current_batch, step, beta=0.1, lambda_fm=10):
        """Inside a Batch"""
        # STEP 1:
        x = data_current_batch.to(settings.device)
        x = torch.reshape(x, (x.shape[0], 1, -1))
        # multi band decomposition pqmf
        #x = self.multi_band_decomposition(x)

        # Encode data
        # if train step 1 repr learning encoder.train()
        # else (train step 2 adversarial) freeze encoder encoder.eval()
        if self.warmed_up:
            self.encoder.eval()
        else:
            self.encoder.train()
        mean, var = self.encoder(x)

        # get latent space samples
        z = self.sampling(x, mean, var)
        # compute regularization loss
        kl_loss = self.kl_div_loss(x, mean, var)
        if self.warmed_up:
            z = z.detach()
            kl_loss = kl_loss.detach()

        # Decode latent space samples
        y = self.decoder(z)

        # compute reconstruction loss ie. multiscale spectral distance
        # spectral_loss = self.multiscale_spectral_distance(x, y)
        spectral_loss = self.spectral_dist_criterion(x, y)
        # total loss
        loss_vae = torch.mean(spectral_loss - beta * kl_loss)

        # inverse multi band decomposition (pqmf -1) --> recomposition
        #x = self.multi_band_decomposition.inverse(x)
        #y = self.multi_band_decomposition.inverse(x_pred)
        #spectral_loss += self.multiscale_spectral_distance(x, y)  # WHY ???

        # STEP 2:
        loss_feature_matching_distance = 0
        loss_adv = 0
        loss_disc = 0
        if self.warmed_up:
            # compute discriminator loss on fake and real data
            real_features = self.discriminator(x)
            fake_features = self.discriminator(y)

            for feat_real, feat_fake in zip(real_features, fake_features):
                # Compute Feature matching distance
                loss_feature_matching_distance += lambda_fm * self.feat_matching_criterion(feat_real, feat_fake).item()

                # Compute Hinge loss
                current_disc_fake_loss = self.disc_criterion(feat_fake[-1], -torch.ones_like(feat_fake[-1]))
                current_disc_real_loss = self.disc_criterion(feat_real[-1], torch.ones_like(feat_real[-1]))
                current_disc_loss = (current_disc_fake_loss + current_disc_real_loss) / 2
                loss_disc += current_disc_loss.item()
                current_gen_loss = self.gen_criterion(feat_fake[-1], -torch.ones_like(feat_fake[-1]))
                loss_adv += current_gen_loss.item()

        # FINALLY:
        # compute vae (decoder-generator) loss
        loss_gen = loss_feature_matching_distance + loss_adv
        loss_vae_gen = loss_vae + loss_gen

        # optimizer steps:
        # Before the backward pass, zero all of the network gradients
        # self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        # loss.backward()
        # Calling the step function to update the parameters
        # self.optimizer.step()
        if step % 2 and self.warmed_up:
            self.disc_opt.zero_grad()
            loss_disc.backward()
            self.disc_opt.ste()
        else:
            self.vae_opt.zero_grad()
            loss_vae_gen.backward()
            self.vae_opt.step()
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

        self._init_optimizer(lr)
        self._init_criterion()

        train_losses = []
        valid_losses = []
        it = 0  # number of batch iterations updated at the end of the DataLoader for loop
        for epoch in range(n_epochs):
            cur_step = 0
            it_display = 0
            loss_display = 0
            for x, _ in tqdm(train_loader):
                step = len(train_loader) * epoch + cur_step
                self.train_step(x, step)

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
