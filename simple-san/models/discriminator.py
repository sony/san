import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDiscriminator(nn.Module):
    def __init__(self, num_class=10):
        super(BaseDiscriminator, self).__init__()

        # Feature extractor
        self.h_function = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=6, stride=2),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
            )

        # Last linear layer
        self.use_class = num_class > 0
        self.fc_w = nn.Parameter(
            torch.randn(num_class if self.use_class else 1, 512 * 3 * 3))

    def forward(self, x, class_ids, flg_train: bool):
        h_feature = self.h_function(x)
        h_feature = torch.flatten(h_feature, start_dim=1)
        weights = self.fc_w[class_ids] if self.use_class else self.fc_w
        out = (h_feature * weights).sum(dim=1)

        return out


# Modified Discriminator Architecture

class SanDiscriminator(BaseDiscriminator):
    def __init__(self, num_class=10):
        super(SanDiscriminator, self).__init__(num_class)

    def forward(self, x, class_ids, flg_train: bool):
        h_feature = self.h_function(x)
        h_feature = torch.flatten(h_feature, start_dim=1)
        weights = self.fc_w[class_ids] if self.use_class else self.fc_w
        direction = F.normalize(weights, dim=1) # Normalize the last layer
        scale = torch.norm(weights, dim=1).unsqueeze(1)
        h_feature = h_feature * scale # For keep the scale
        if flg_train: # for discriminator training
            out_fun = (h_feature * direction.detach()).sum(dim=1)
            out_dir = (h_feature.detach() * direction).sum(dim=1)
            out = dict(fun=out_fun, dir=out_dir)
        else: # for generator training or inference
            out = (h_feature * direction).sum(dim=1)

        return out
