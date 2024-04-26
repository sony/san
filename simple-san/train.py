import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

from models.discriminator import BaseDiscriminator, SanDiscriminator
from models.generator import Generator
from torch.utils.data import DataLoader


def update_discriminator(x, class_ids, discriminator, generator, optimizer, params):
    bs = x.size(0)
    device = x.device

    optimizer.zero_grad()

    # for data (ground-truth) distribution
    disc_real = discriminator(x, class_ids, flg_train=True)
    loss_real = eval('compute_loss_'+args.model)(disc_real, loss_type='real')

    # for generator distribution
    latent = torch.randn(bs, params["dim_latent"], device=device)
    img_fake = generator(latent, class_ids)
    disc_fake = discriminator(img_fake.detach(), class_ids, flg_train=True)
    loss_fake = eval('compute_loss_'+args.model)(disc_fake, loss_type='fake')


    loss_d = loss_real + loss_fake
    loss_d.backward()
    optimizer.step()


def update_generator(num_class, discriminator, generator, optimizer, params, device):
    optimizer.zero_grad()

    bs = params['batch_size']
    latent = torch.randn(bs, params["dim_latent"], device=device)

    class_ids = torch.randint(num_class, size=(bs,), device=device)
    batch_fake = generator(latent, class_ids)

    disc_gen = discriminator(batch_fake, class_ids, flg_train=False)
    loss_g = - disc_gen.mean()
    loss_g.backward()
    optimizer.step()


def compute_loss_gan(disc, loss_type):
    assert (loss_type in ['real', 'fake'])
    if 'real' == loss_type:
        loss = (1. - disc).relu().mean() # Hinge loss
    else: # 'fake' == loss_type
        loss = (1. + disc).relu().mean() # Hinge loss

    return loss


def compute_loss_san(disc, loss_type):
    assert (loss_type in ['real', 'fake'])
    if 'real' == loss_type:
        loss_fun = (1. - disc['fun']).relu().mean() # Hinge loss for function h
        loss_dir = - disc['dir'].mean() # Wasserstein loss for omega
    else: # 'fake' == loss_type
        loss_fun = (1. + disc['fun']).relu().mean() # Hinge loss for function h
        loss_dir = disc['dir'].mean() # Wasserstein loss for omega
    loss = loss_fun + loss_dir

    return loss


def save_images(imgs, idx, dirname='test'):
    import numpy as np
    if imgs.shape[1] == 1:
        imgs = np.repeat(imgs, 3, axis=1)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(imgs):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)))

    if not os.path.exists('out/{}/'.format(dirname)):
        os.makedirs('out/{}/'.format(dirname))
    plt.savefig('out/{0}/{1}.png'.format(dirname, str(idx).zfill(3)), bbox_inches="tight")
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True, help="path to MNIST dataset folder")
    parser.add_argument("--params", type=str, default="./hparams/params.json", help="path to hyperparameters")
    parser.add_argument("--model", type=str, default="gan", help="model's name / 'gan' or 'san'")
    parser.add_argument('--enable_class', action='store_true', help='enable class conditioning')
    parser.add_argument("--logdir", type=str, default="./logs", help="directory storing log files")
    parser.add_argument("--device", type=int, default=0, help="gpu device to use")

    return parser.parse_args()


def main(args):
    with open(args.params, "r") as f:
        params = json.load(f)

    device = f'cuda:{args.device}' if args.device is not None else 'cpu'
    model_name = args.model
    if not model_name in ['gan', 'san']:
        raise RuntimeError("A model name have to be 'gan' or 'san'.")
    experiment_name = model_name + "_cond" if args.enable_class else model_name

    # dataloading
    num_class = 10
    train_dataset = datasets.MNIST(root=args.datadir, transform=transforms.ToTensor(), train=True, download=True)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], num_workers=4,
                              pin_memory=True, persistent_workers=True, shuffle=True)
    test_dataset = datasets.MNIST(root=args.datadir, transform=transforms.ToTensor(), train=False, download=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], num_workers=4,
                             pin_memory=True, persistent_workers=True, shuffle=False)

    # model
    use_class = args.enable_class
    generator = Generator(params["dim_latent"], num_class=num_class if use_class else 0)
    if 'gan' == args.model:
        discriminator = BaseDiscriminator(num_class=num_class if use_class else 0)
    else: # 'san' == args.model
        discriminator = SanDiscriminator(num_class=num_class if use_class else 0)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # optimizer
    betas = (params["beta_1"], params["beta_2"])
    optimizer_G = optim.Adam(generator.parameters(), lr=params["learning_rate"], betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params["learning_rate"], betas=betas)

    ckpt_dir = f'{args.logdir}/{experiment_name}/'
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    steps_per_epoch = len(train_loader)

    msg = ["\t{0}: {1}".format(key, val) for key, val in params.items()]
    print("hyperparameters: \n" + "\n".join(msg))

    # eval initial states
    num_samples_per_class = 10
    with torch.no_grad():
        latent = torch.randn(num_samples_per_class * num_class, params["dim_latent"]).cuda()
        class_ids = torch.arange(num_class, dtype=torch.long,
                                 device=device).repeat_interleave(num_samples_per_class)
        imgs_fake = generator(latent, class_ids)

    # main training loop
    for n in range(params["num_epochs"]):
        loader = iter(train_loader)

        print("epoch: {0}/{1}".format(n + 1, params["num_epochs"]))
        for i in tqdm.trange(steps_per_epoch):
            x, class_ids = next(loader)
            x = x.to(device)
            class_ids = class_ids.to(device)

            update_discriminator(x, class_ids, discriminator, generator, optimizer_D, params)
            update_generator(num_class, discriminator, generator, optimizer_G, params, device)

        torch.save(generator.state_dict(), ckpt_dir + "g." + str(n) + ".tmp")
        torch.save(discriminator.state_dict(), ckpt_dir + "d." + str(n) + ".tmp")

        # eval
        with torch.no_grad():
            latent = torch.randn(num_samples_per_class * num_class, params["dim_latent"]).cuda()
            class_ids = torch.arange(num_class, dtype=torch.long,
                                     device=device).repeat_interleave(num_samples_per_class)
            imgs_fake = generator(latent, class_ids).cpu().data.numpy()
            save_images(imgs_fake, n, dirname=experiment_name)
    
    torch.save(generator.state_dict(), ckpt_dir + "generator.pt")
    torch.save(discriminator.state_dict(), ckpt_dir + "discriminator.pt")


if __name__ == '__main__':
    args = get_args()
    main(args)
