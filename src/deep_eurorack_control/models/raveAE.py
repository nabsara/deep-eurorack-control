import os
import torch
import torch.distributions as distributions
import numpy as np
from tqdm import tqdm
import time
import random
from torch.utils.tensorboard import SummaryWriter

from deep_eurorack_control.config import settings
from deep_eurorack_control.models.networks import Decoder, Discriminator
from deep_eurorack_control.models.networks.encoderAE import EncoderAE
from deep_eurorack_control.models.networks.pqmf_antoine import PQMF
from deep_eurorack_control.models.losses import SpectralLoss, LinearLoss, HingeLoss, FeatureMatchingLoss
from deep_eurorack_control.models.fader_discriminator import FaderLoss, FaderDiscriminator


class RaveAE:

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
        self.encoder = EncoderAE(
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

        self.fader_discriminator = FaderDiscriminator().to(settings.device)

        self.warmed_up = False

    def _init_optimizer(self, learning_rate, beta_1=0.5, beta_2=0.9):
        param_ae = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.ae_opt = torch.optim.Adam(param_ae, lr=learning_rate, betas=(beta_1, beta_2))
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
        self.fader_disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

    def _init_criterion(self):
        self.spectral_dist_criterion = SpectralLoss()
        self.disc_criterion = HingeLoss()
        self.gen_criterion = LinearLoss()
        self.feat_matching_criterion = FeatureMatchingLoss()
        self.fader_criterion = FaderLoss()

    def train_step(self, data_current_batch, params_current_batch, step, beta=0.1, lambda_fm=10):
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
            self.encoder.eval() # freeze?
        else:
            self.encoder.train()
        self.decoder.train()
        z = self.encoder(x)

        if self.warmed_up:
            z = z.detach()

        # Decode latent space samples
        y = self.decoder(z)

        # compute reconstruction loss ie. multiscale spectral distance
        spectral_loss = self.spectral_dist_criterion(x, y)
        # total loss
        loss_ae = torch.mean(spectral_loss)

        # inverse multi band decomposition (pqmf -1) --> recomposition
        x = self.multi_band_decomposition.inverse(x)
        y = self.multi_band_decomposition.inverse(y)
        spectral_loss += self.spectral_dist_criterion(x, y)  # WHY ???

        # encode
        # lat dis z res1
        # lat dis y 
        z_fader = self.fader_discriminator(z)

        # compute loss
        #attributes = []
        #keys = []
        #key_indices = random.sample(keys, 8) # if batch_size = 8
        #f_style_cls = []
        #class_label = torch.LongTensor([[f_style_cls[attr + '/' + key] for attr in attributes] for key in key_indices], settings.device).reshape(8, -1)

        #loss_fader = self.fader_criterion(z_fader, class_label)

        # compute loss
        class_label = torch.LongTensor(params_current_batch, settings.device).reshape(8, -1)

        loss_fader = self.fader_criterion(z_fader, class_label)


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
        # compute ae (decoder-generator) loss
        loss_gen = loss_feature_matching_distance + loss_adv
        loss_total = loss_ae + loss_gen + loss_fader

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
            self.ae_opt.zero_grad()
            loss_total.backward(retain_graph=True)
            self.ae_opt.step()

        losses = {
            "loss_ae": loss_ae.item(),
            "loss_multiscale_spectral_dist": spectral_loss.item(),
            "loss_gen": loss_gen.item(),
            "loss_feature_matching_dist": loss_feature_matching_distance.item(),
            "loss_adv": loss_adv.item(),
            "loss_total_ae_gen": loss_total.item(),
            "loss_disc": loss_disc.item(),
            "loss_fader_disc": loss_fader.item(),
        }
        return losses

    def validation_step(self, data_current_batch, beta=0.1,):
        # For AE training step 1 repr learning
        # model.eval() mode
        self.encoder.eval()
        self.decoder.eval()

        x = data_current_batch.to(settings.device)
        x = torch.reshape(x, (x.shape[0], 1, -1))
        # 1. multi band decomposition pqmf
        x = self.multi_band_decomposition(x)

        # 2. Encode data
        z = self.encoder(x)
        z_fader = self.fader_discriminator(z)

        # 3. Decode latent space samples
        y = self.decoder(z)

        # 4. inverse multi band decomposition (pqmf -1) --> recomposition
        x = self.multi_band_decomposition.inverse(x)
        y = self.multi_band_decomposition.inverse(y)

        # 5. compute reconstruction loss ie. multiscale spectral distance
        spectral_distance = torch.mean(self.spectral_dist_criterion(x, y))

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
            "loss_ae": [],
            "loss_multiscale_spectral_dist": [],
            "loss_gen": [],
            "loss_feature_matching_dist": [],
            "loss_adv": [],
            "loss_total_ae_gen": [],
            "loss_disc": [],
            "loss_fader_disc": [], 
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
            for x, _, attr in tqdm(train_loader):
                step = len(train_loader) * epoch + cur_step
                cur_losses = self.train_step(x, attr, step)

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
                        f"\tTime: {time.time() - start} (s) \tTotal_loss: {train_losses['loss_total_ae_gen'][-1]}"
                    )
                    losses_display = np.zeros(len(train_losses.keys()) - 2)
                    it_display = 0
                cur_step += 1
                it += 1

            with torch.no_grad():
                for x, _ in tqdm(valid_loader):
                    x = x.to(settings.device)
                    y, loss = self.validation_step(x)

                    valid_loss_display += loss.item()
                    # add audio to tensorboard
                    if it_display == 0:
                        for j in range(x.shape[0]):
                            writer.add_audio(
                                "generated_sound/" + str(j),
                                y,
                                global_step=epoch * len(valid_loader) + cur_step,
                                sample_rate=self.sampling_rate,
                            )
                            writer.add_audio(
                                "ground_truth_sound/" + str(j),
                                x,
                                global_step=epoch * len(valid_loader) + cur_step,
                                sample_rate=self.sampling_rate,
                            )
                    it_display += 1
                print(
                    f"\nEpoch: [{epoch}/{n_epochs}] \t Validation loss: {valid_loss_display / it_display}"
                )
                writer.add_scalar(
                    f"Validation total loss :",
                    valid_loss_display / it_display,
                    epoch * len(valid_loader) + cur_step,
                )
                valid_loss.append(valid_loss_display / it_display)

            # model checkpoints:
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "optimizer_state_dict": self.ae_opt.state_dict(),
                        "loss": train_losses['loss_total_ae_gen'][-1]
                    },
                    os.path.join(models_dir, f"{model_filename}__ae.pt")
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
