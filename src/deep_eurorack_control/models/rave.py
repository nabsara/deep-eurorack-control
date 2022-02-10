import os
import torch
import torch.distributions as distributions
import numpy as np
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

from deep_eurorack_control.config import settings
from deep_eurorack_control.models.networks import Encoder, Decoder, Discriminator
from deep_eurorack_control.models.networks.pqmf_antoine import PQMF
from deep_eurorack_control.models.losses import SpectralLoss, LinearLoss, HingeLoss, FeatureMatchingLoss
from deep_eurorack_control.helpers.data_viz import plot_metrics


class RAVE:

    def __init__(self, n_band=16, latent_dim=128, hidden_dim=64, sampling_rate=16000):

        self.model_name = "n_synth_rave"
        self.sampling_rate = sampling_rate

        # n_taps=4
        # self.multi_band_decomposition = PQMF(
        #    n_band=n_band,
        #    n_taps=n_taps,
        #    sampling_rate=sampling_rate
        # )
        self.multi_band_decomposition = PQMF(
            attenuation=100,
            n_band=n_band,
            polyphase=False
        )

        data_size = n_band
        self.encoder = Encoder(
            data_size=data_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(settings.device)
        self.decoder = Decoder(
            data_size=data_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        ).to(settings.device)
        self.discriminator = Discriminator().to(settings.device)

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
        epsilon = q.sample((int(n_batch),)).to(settings.device)
        # TODO: test
        # epsilon = torch.randn_like(mu)
        z = mu + sigma * epsilon.unsqueeze(2)
        return z

    @staticmethod
    def kl_div_loss(mu, sigma):
        # kl_div = torch.mean(-0.5 * torch.sum(1 + sigma - torch.pow(mu, 2) - torch.exp(sigma), dim=1))
        kl_div = torch.mean(-0.5 * torch.sum(1 - sigma - torch.pow(mu, 2) + torch.log(sigma), dim=1))
        return kl_div

    def reparametrize(self, mean, scale):
        std = torch.nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        return z, kl

    def train_step(self, data_current_batch, step, beta=0.1, lambda_fm=10):
        """Inside a Batch"""
        # STEP 1:
        x = data_current_batch.to(settings.device)
        x = torch.reshape(x, (x.shape[0], 1, -1))
        # multi band decomposition pqmf
        x = self.multi_band_decomposition(x)

        # Encode data
        # if train step 1 repr learning encoder.train()
        # else (train step 2 adversarial) freeze encoder encoder.eval()
        if self.warmed_up:
            self.encoder.eval()
        else:
            self.encoder.train()
        self.decoder.train()
        mean, var = self.encoder(x)

        # get latent space samples
        #z = self.sampling(x, mean, var)
        # compute regularization loss
        #kl_loss = self.kl_div_loss(mean, var)
        z, kl_loss = self.reparametrize(mean, var)
        if self.warmed_up:
            z = z.detach()
            kl_loss = kl_loss.detach()

        # Decode latent space samples
        y = self.decoder(z)

        # compute reconstruction loss ie. multiscale spectral distance
        spectral_loss = self.spectral_dist_criterion(x, y)
        # total loss
        loss_vae = torch.mean(spectral_loss + beta * kl_loss)

        # inverse multi band decomposition (pqmf -1) --> recomposition
        x = self.multi_band_decomposition.inverse(x)
        y = self.multi_band_decomposition.inverse(y)
        spectral_loss += self.spectral_dist_criterion(x, y)  # WHY ???

        # STEP 2:
        if self.warmed_up:
            self.discriminator.train()
            # compute discriminator loss on fake and real data
            real_features = self.discriminator(x)
            fake_features = self.discriminator(y)

            loss_feature_matching_distance = 0
            loss_adv = 0
            loss_disc = 0
            for feat_real, feat_fake in zip(real_features, fake_features):
                # Compute Feature matching distance
                loss_feature_matching_distance += lambda_fm * self.feat_matching_criterion(feat_real, feat_fake)

                # Compute Hinge loss
                current_disc_fake_loss = self.disc_criterion(feat_fake[-1], -torch.ones_like(feat_fake[-1]))
                current_disc_real_loss = self.disc_criterion(feat_real[-1], torch.ones_like(feat_real[-1]))
                current_disc_loss = (current_disc_fake_loss + current_disc_real_loss) / 2
                loss_disc += current_disc_loss
                current_gen_loss = self.gen_criterion(feat_fake[-1], -torch.ones_like(feat_fake[-1]))
                loss_adv += current_gen_loss
        else:
            loss_feature_matching_distance = torch.tensor(0.).to(x)
            loss_disc = torch.tensor(0.).to(x)
            loss_adv = torch.tensor(0.).to(x)

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
            loss_disc.backward(retain_graph=True)
            self.disc_opt.step()
        else:
            self.vae_opt.zero_grad()
            loss_vae_gen.backward(retain_graph=True)
            self.vae_opt.step()

        losses = {
            "loss_vae": loss_vae.item(),
            "loss_multiscale_spectral_dist": spectral_loss.item(),
            "loss_kl_div": kl_loss.item(),
            "loss_gen": loss_gen.item(),
            "loss_feature_matching_dist": loss_feature_matching_distance.item(),
            "loss_adv": loss_adv.item(),
            "loss_total_vae_gen": loss_vae_gen.item(),
            "loss_disc": loss_disc.item(),
        }
        return losses

    def validation_step(self, data_current_batch, beta=0.1,):
        # For VAE training step 1 repr learning
        # model.eval() mode
        self.encoder.eval()
        self.decoder.eval()

        x = data_current_batch.to(settings.device)
        x = torch.reshape(x, (x.shape[0], 1, -1))
        # 1. multi band decomposition pqmf
        x = self.multi_band_decomposition(x)

        # 2. Encode data
        mean, var = self.encoder(x)

        # 3. get latent space samples
        #z = self.sampling(x, mean, var)
        # compute regularization loss
        #kl_loss = self.kl_div_loss(mean, var)
        z, kl_loss = self.reparametrize(mean, var)

        # 4. Decode latent space samples
        y = self.decoder(z)

        # 5. inverse multi band decomposition (pqmf -1) --> recomposition
        x = self.multi_band_decomposition.inverse(x)
        y = self.multi_band_decomposition.inverse(y)

        # 6. compute reconstruction loss ie. multiscale spectral distance
        spectral_distance = torch.mean(self.spectral_dist_criterion(x, y) + beta * kl_loss)

        return y, spectral_distance

    def train(self, train_loader, valid_loader, lr, n_epochs, display_step, models_dir, model_filename, n_epoch_warmup):
        start = time.time()

        # TENSORBOARD
        writer = SummaryWriter(
            os.path.join(
                models_dir,
                f"runs/exp__{model_filename}_{time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime())}",
            )
        )

        self._init_optimizer(lr)
        self._init_criterion()

        train_losses = {
            "it": [],
            "epoch": [],
            "loss_vae": [],
            "loss_multiscale_spectral_dist": [],
            "loss_kl_div": [],
            "loss_gen": [],
            "loss_feature_matching_dist": [],
            "loss_adv": [],
            "loss_total_vae_gen": [],
            "loss_disc": [],
        }
        valid_loss = []
        it = 0  # number of batch iterations updated at the end of the DataLoader for loop
        for epoch in range(n_epochs):
            if epoch == n_epoch_warmup:
                self.warmed_up = True
            cur_step = 0
            it_display = 0
            valid_loss_display = 0
            losses_display = np.zeros(len(train_losses.keys()) - 2)
            for x, _ in tqdm(train_loader):
                step = len(train_loader) * epoch + cur_step
                cur_losses = self.train_step(x, step)

                # keep track of the loss
                losses_display += np.asarray(list(cur_losses.values()))
                it_display += 1
                if it % display_step == 0 or (
                        (epoch == n_epochs - 1) and (cur_step == len(train_loader) - 1)
                ):
                    train_losses['it'].append(it)
                    train_losses['epoch'].append(epoch)
                    for k, l in zip(list(cur_losses.keys()), losses_display):
                        train_losses[k].append(l / it_display)
                        writer.add_scalar(
                            f"training loss : {k}",
                            l / it_display,
                            epoch * len(train_loader) + cur_step,
                        )

                    print(
                        f"\nEpoch: [{epoch}/{n_epochs}] \tStep: [{cur_step}/{len(train_loader)}]"
                        f"\tTime: {time.time() - start} (s) \tTotal_loss: {train_losses['loss_total_vae_gen'][-1]}"
                    )
                    losses_display = np.zeros(len(train_losses.keys()) - 2)
                    it_display = 0
                cur_step += 1
                it += 1

            with torch.no_grad():
                x_eval = []
                y_eval = []
                nb_examples = 10
                for x, _ in tqdm(valid_loader):
                    x = x.to(settings.device)
                    y, loss = self.validation_step(x)
                    if len(x_eval) < nb_examples:
                        x_eval.append(np.squeeze(x[0].detach().cpu().numpy()))
                        y_eval.append(np.squeeze(y[0].detach().cpu().numpy()))
                    valid_loss_display += loss.item()
                    it_display += 1
                print(
                    f"\nEpoch: [{epoch}/{n_epochs}] \t Validation loss: {valid_loss_display / it_display}"
                )
                writer.add_scalar(
                    f"Validation total loss :",
                    valid_loss_display / it_display,
                    epoch * len(valid_loader) + cur_step,
                )
                # add audio to tensorboard
                for j in range(len(x_eval)):
                    fig_stft = plot_metrics(signal_in=x_eval[j], signal_out=y_eval[j], sr=self.sampling_rate)
                    writer.add_audio(
                        "ground_truth_sound/" + str(j),
                        x_eval[j],
                        global_step=epoch * len(valid_loader) + cur_step,
                        sample_rate=self.sampling_rate,
                    )
                    writer.add_audio(
                        "generated_sound/" + str(j),
                        y_eval[j],
                        global_step=epoch * len(valid_loader) + cur_step,
                        sample_rate=self.sampling_rate,
                    )
                    writer.add_figure(
                        "Output Images/" + str(j),
                        fig_stft,
                        global_step=epoch * len(valid_loader) + cur_step,
                    )
                valid_loss.append(valid_loss_display / it_display)

            # model checkpoints:
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "optimizer_state_dict": self.vae_opt.state_dict(),
                        "loss": train_losses['loss_total_vae_gen'][-1]
                    },
                    os.path.join(models_dir, f"{model_filename}__vae.pt")
                )
                if self.warmed_up:
                    torch.save(
                        {
                            "epoch": epoch,
                            "discriminator_state_dict": self.discriminator.state_dict(),
                            "optimizer_state_dict": self.disc_opt.state_dict(),
                            "loss": valid_loss
                        },
                        os.path.join(models_dir, f"{model_filename}__discriminator.pt")
                    )
