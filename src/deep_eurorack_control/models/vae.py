import os
import time
import torch
import torch.nn as nn
import torch.distributions as distributions
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import tqdm

from deep_eurorack_control.config import settings


def show_img(img_truth, img_reconstruction, n_img=16):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    image_grid_truth = make_grid(img_truth.detach().cpu()[:n_img], nrow=4)
    ax1.imshow(image_grid_truth.permute(1, 2, 0).squeeze())
    image_grid_reconstruction = make_grid(img_reconstruction.detach().cpu()[:n_img], nrow=4)
    ax2.imshow(image_grid_reconstruction.permute(1, 2, 0).squeeze())
    plt.show()


class LinearVAE(nn.Module):
    def __init__(self, img_width=28, img_height=28, latent_dim=16, hidden_dim=512, img_channel=1):
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.img_size = img_width * img_height
        self.img_channel = img_channel

        self.z_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.img_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.z_dim),
            nn.ReLU()
        )
        self.sigma = nn.Sequential(
            nn.Linear(self.hidden_dim, self.z_dim),
            nn.Softplus()  # SoftPlus is a smooth approximation to the
            # ReLU function and can be used to constrain the output of
            # a machine to always be positive. (cf. torch docs)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.img_size * self.img_channel),
            nn.Sigmoid()
        )

    @staticmethod
    def sampling(x, mu, sigma):
        n_batch = x.shape[0]
        q = distributions.Normal(torch.zeros(mu.shape[1]), torch.ones(sigma.shape[1]))
        epsilon = q.sample((int(n_batch), ))
        z = mu + sigma * epsilon
        return z

    # NB: The loss is the BCE loss combined with the KL divergence
    @staticmethod
    def kl_div_loss(x, mu, sigma):
        n_batch = x.shape[0]
        kl_div = 0.5 * torch.sum(1 + sigma - torch.pow(mu, 2) - torch.exp(sigma))
        return kl_div / n_batch

    def forward(self, x):
        # encode the inputs
        x_enc = self.encoder(x)
        mu = self.mu(x_enc)
        sigma = self.sigma(x_enc)

        # get latent space samples
        z_tilde = self.sampling(x, mu, sigma)

        # compute KL divergence
        kl_div = self.kl_div_loss(x, mu, sigma)

        # decode the samples
        x_tilde = self.decoder(z_tilde).reshape(-1, self.img_channel, self.img_width, self.img_height)

        return x_tilde, kl_div


class ModelVAE:
    def __init__(self):
        self.model = LinearVAE()
        self.model_name = "mnist_linear_vae"

    def _init_optimizer(self, learning_rate, beta_1=0.5, beta_2=0.9):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

    def _init_reconstruction_criterion(self):
        self.criterion = torch.nn.MSELoss(reduction="sum")

    def train(self, train_loader, valid_loader, lr, n_epochs, display_step, models_dir, model_filename, show_fig):

        start = time.time()

        # TENSORBOARD
        writer = SummaryWriter(
            os.path.join(
                models_dir,
                f"runs/exp__{self.model_name}__lr_{lr}__e_{n_epochs}_"
                + time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()),
            )
        )

        self._init_optimizer(lr)
        self._init_reconstruction_criterion()

        train_losses = []
        valid_losses = []
        it = 0  # number of batch iterations updated at the end of the DataLoader for loop
        for epoch in range(n_epochs):
            cur_step = 0
            it_display = 0
            loss_display = 0
            self.model.train()
            for x, _ in tqdm(train_loader):
                x = x.to(settings.device)

                # model predictions and compute regularisation loss
                x_tilde, kl_div = self.model(x)
                # compute reconstruction loss
                cross_entropy = self.criterion(x_tilde, x)
                # compute total loss
                loss = torch.mean(cross_entropy - kl_div)

                # Before the backward pass, zero all of the network gradients
                self.optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to parameters
                loss.backward()
                # Calling the step function to update the parameters
                self.optimizer.step()

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

                    writer.add_scalar(
                        "training loss",
                        loss_display / it_display,
                        epoch * len(train_loader) + cur_step,
                    )

                    train_losses.append(loss_display / it_display)
                    loss_display = 0
                    it_display = 0
                cur_step += 1
                it += 1

            self.model.eval()
            with torch.no_grad():
                for x, _ in tqdm(valid_loader):
                    x = x.to(settings.device)
                    # model predictions and compute regularisation loss
                    x_tilde, kl_div = self.model(x)
                    # compute reconstruction loss
                    cross_entropy = self.criterion(x_tilde, x)
                    # compute total loss
                    loss = torch.mean(cross_entropy - kl_div)

                    loss_display += loss.item()
                    it_display += 1

                print(
                    f"\nEpoch: [{epoch}/{n_epochs}] \t Validation loss: {loss_display / it_display}"
                )
                writer.add_scalar(
                    "validation loss",
                    loss_display / it_display,
                    epoch * len(valid_loader) + cur_step,
                )
                writer.add_image(
                    "images ground truth",
                    make_grid(x.detach().cpu()[:16], nrow=4),
                    epoch * len(valid_loader) + cur_step,
                )
                writer.add_image(
                    "images reconstruction",
                    make_grid(x_tilde.detach().cpu()[:16], nrow=4),
                    epoch * len(valid_loader) + cur_step,
                )

                if show_fig:
                    show_img(x, x_tilde)
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
