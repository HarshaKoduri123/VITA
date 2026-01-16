# cvae.py
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalVAE(nn.Module):


    def __init__(
        self,
        in_channels_x: int,
        image_shape: int,
        latent_dim: int = 32,
        hidden_dims=None,
        beta: float = 1.0,
    ):
        super().__init__()
        self.in_channels_x = in_channels_x
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]


        enc_in = in_channels_x + 1  # x channels + heatmap channel
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_in, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(h_dim),
                    nn.ReLU(inplace=True),
                )
            )
            enc_in = h_dim
        self.encoder = nn.Sequential(*modules)


        down_factor = 2 ** len(hidden_dims)
        final_spatial = max(1, image_shape // down_factor)
        self.final_spatial = final_spatial
        self.final_channels = hidden_dims[-1]

        self.fc_mu = nn.Conv3d(self.final_channels, latent_dim, kernel_size=1)
        self.fc_var = nn.Conv3d(self.final_channels, latent_dim, kernel_size=1)

       
        cond_modules = []
        cond_in = in_channels_x
        for h_dim in hidden_dims:
            cond_modules.append(
                nn.Sequential(
                    nn.Conv3d(cond_in, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(h_dim),
                    nn.ReLU(inplace=True),
                )
            )
            cond_in = h_dim
        self.cond_down = nn.Sequential(*cond_modules)

  
        self.decoder_input = nn.Sequential(
            nn.Conv3d(latent_dim + self.final_channels, self.final_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(self.final_channels),
            nn.ReLU(inplace=True),
        )

        dec_hidden = list(reversed(hidden_dims))
        up_modules = []
        for i in range(len(dec_hidden) - 1):
            up_modules.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        dec_hidden[i],
                        dec_hidden[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(dec_hidden[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )
        self.decoder = nn.Sequential(*up_modules)

        self.final_layer = nn.Sequential(
            nn.Conv3d(dec_hidden[-1], 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, H):
        inp = torch.cat([x, H], dim=1)
        h = self.encoder(inp)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x):
        x_down = self.cond_down(x)
        zx = torch.cat([z, x_down], dim=1)
        h = self.decoder_input(zx)
        h = self.decoder(h)


        H_hat = self.final_layer(h)
        if H_hat.shape[-1] != x.shape[-1]:
            H_hat = F.interpolate(H_hat, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return H_hat

    def forward(self, x, H):
        mu, log_var = self.encode(x, H)
        z = self.reparameterize(mu, log_var)
        H_hat = self.decode(z, x)
        return H_hat, mu, log_var

    @torch.no_grad()
    def sample(self, x, num_samples: int = None):

        B, _, D, H, W = x.shape
        if num_samples is None:
            num_samples = B

        z = torch.randn((B, self.latent_dim, self.final_spatial, self.final_spatial, self.final_spatial), device=x.device)
        # print(z.shape)
        x_down = self.cond_down(x)
        zx = torch.cat([z, x_down], dim=1)
        h = self.decoder_input(zx)
        h = self.decoder(h)
        H_hat = self.final_layer(h)
        # print(H_hat.shape)
        if H_hat.shape[-1] != D:
            H_hat = F.interpolate(H_hat, size=(D, H, W), mode="trilinear", align_corners=False)
        return H_hat
