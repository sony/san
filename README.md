# Slicing Adversarial Network (SAN) [ICLR 2024]

This repository contains the official PyTorch implementation of **"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer"** (*[arXiv 2301.12811](https://arxiv.org/abs/2301.12811)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

### [[Project Page]](https://ytakida.github.io/san/)


> **Abstract:** Generative adversarial networks (GANs) learn a target probability distribution by optimizing a generator and a discriminator with minimax objectives. This paper addresses the question of whether such optimization actually provides the generator with gradients that make its distribution close to the target distribution. We derive metrizable conditions, sufficient conditions for the discriminator to serve as the distance between the distributions, by connecting the GAN formulation with the concept of sliced optimal transport. Furthermore, by leveraging these theoretical results, we propose a novel GAN training scheme called the Slicing Adversarial Network (SAN). With only simple modifications, a broad class of existing GANs can be converted to SANs. Experiments on synthetic and image datasets support our theoretical results and the effectiveness of SAN as compared to the usual GANs. We also apply SAN to StyleGAN-XL, which leads to a state-of-the-art FID score amongst GANs for class conditional generation on CIFAR10 and ImageNet 256$times$256. 


# Citation
[1] Takida, Y., Imaizumi, M., Shibuya, T., Lai, C., Uesaka, T., Murata, N. and Mitsufuji, Y.,
"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer,"
ICLR 2024.
```
@inproceedings{takida2024san,
    title={{SAN}: Inducing Metrizability of {GAN} with Discriminative Normalized Linear Layer},
    author={Takida, Yuhta and Imaizumi, Masaaki and Shibuya, Takashi and Lai, Chieh-Hsin and Uesaka, Toshimitsu and Murata, Naoki and Mitsufuji, Yuki},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=eiF7TU1E8E}
}
```