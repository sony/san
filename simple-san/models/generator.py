import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dim_latent=100, num_class=10):
        super(Generator, self).__init__()

        self.dim_latent = dim_latent
        self.use_class = num_class > 0

        if self.use_class:
            self.emb_class = nn.Embedding(num_class, dim_latent)
            self.fc = nn.Linear(dim_latent * 2, 512 * 3 * 3)
        else:
            self.fc = nn.Linear(dim_latent, 512 * 3 * 3)

        self.g_function = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=6, stride=2),
            nn.Sigmoid()
            )

    def forward(self, x, class_ids):
        batch_size = x.size(0)

        if self.use_class:
            x_class = self.emb_class(class_ids)
            x = self.fc(torch.cat((x, x_class), dim=1))
        else:
            x = self.fc(x)

        x = F.leaky_relu(x)
        x = x.view(batch_size, 512, 3, 3)
        img = self.g_function(x)

        return img
