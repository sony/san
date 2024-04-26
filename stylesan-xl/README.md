# Slicing Adversarial Network (SAN) [ICLR 2024]

This repository contains the official PyTorch implementation of **"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer"** (*[arXiv 2301.12811](https://arxiv.org/abs/2301.12811)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

### [[Project Page]](https://ytakida.github.io/san/)


## Requirements
This repository builds on the codebase of [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl).

- 64-bit Python 3.8 and PyTorch 2.1.1 (or later). See https://pytorch.org for PyTorch install instructions.
- CUDA toolkit 12.1 or later.
- GCC 7 or later compilers. The recommended GCC version depends on your CUDA version; see for example, CUDA 11.4 system requirements.
- If you run into problems when setting up the custom CUDA kernels, we refer to the [Troubleshooting docs](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary) of the original StyleGAN3 repo and the following issues: https://github.com/autonomousvision/stylegan_xl/issues/23.
- Windows user struggling installing the env might find https://github.com/autonomousvision/stylegan_xl/issues/10
  helpful.
  
We provide a [Dockerfile](Dockerfile) for Docker users.

## Reproduction on ImageNet
### Data Preparation
You can download the ImageNet dataset [here](https://image-net.org/). To preprocess the dataset, run
```
python dataset_tool_for_imagenet.py --source=<path_to_imagenet>/ILSVRC --dest=./data/imagenet256.zip \
  --resolution=256x256 --transform=center-crop
```

### Training
We trained our StyleSAN-XL model as follows:
```
python train.py --outdir=./training-runs/imagenet --cfg=stylegan3-t --data=./data/imagenet256.zip \
  --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 11000 --syn_layers 11 \
  --superres --cls_weight 8.0 --up_factor 2 --head_layers 7 \
  --path_stem https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet128.pkl \
  --cond True
```

This repository builds on the codebase of [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl). Please refer to the repository for more details.

## Reproduction on CIFAR10
### Data Preparation
You can download the CIFAR10 dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html). To preprocess the dataset, run
```
python dataset_tool.py --source=./data/cifar-10-python.tar.gz --dest=./data/cifar10_16.zip \
  --resolution=16x16

python dataset_tool.py --source=./data/cifar-10-python.tar.gz --dest=./data/cifar10_32.zip \
  --resolution=32x32
```

### Training
```
python train.py --outdir=./training-runs/cifar10 --cfg=stylegan3-r --data=./data/cifar10_16.zip \
        --gpus=8 --batch=2048 --mirror=1 --snap 10 --batch-gpu 8 --kimg 300000 --syn_layers 6 \
        --cond True

python train.py --outdir=./training-runs/cifar10 --cfg=stylegan3-r --data=./data/cifar10_32.zip \
        --gpus=8 --batch=2048 --mirror=1 --snap 10 --batch-gpu 8 --kimg 250000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 7 \
        --path_stem training-runs/cifar10/00000-stylegan3-r-cifar10_16-gpus8-batch2048/best_model.pkl \
        --cond True
```

You can change `--gpus` and `--batch-gpu` arguments according to your environment.

## Reproduction on FFHQ
### Data Preparation
You can download the FFHQ dataset [here](https://github.com/NVlabs/ffhq-dataset). To preprocess the dataset, run
```
python dataset_tool.py --source=./data/ffhq/ --dest=./data/ffhq16.zip \
  --resolution=16x16

python dataset_tool.py --source=./data/ffhq/ --dest=./data/ffhq32.zip \
  --resolution=32x32

python dataset_tool.py --source=./data/ffhq/ --dest=./data/ffhq64.zip \
  --resolution=64x64

python dataset_tool.py --source=./data/ffhq/ --dest=./data/ffhq128.zip \
  --resolution=128x128

python dataset_tool.py --source=./data/ffhq/ --dest=./data/ffhq256.zip \
  --resolution=256x256

python dataset_tool.py --source=./data/ffhq/ --dest=./data/ffhq512.zip \
  --resolution=512x512

python dataset_tool.py --source=./data/ffhq/ --dest=./data/ffhq1024.zip \
  --resolution=1024x1024
```

### Training
```
python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-r --data=./data/ffhq16.zip \
        --gpus=8 --batch=2048 --mirror=1 --snap 10 --batch-gpu 8 --kimg 300000 --syn_layers 6

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-r --data=./data/ffhq32.zip \
        --gpus=8 --batch=2048 --mirror=1 --snap 10 --batch-gpu 8 --kimg 175000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 7 \
        --path_stem training-runs/ffhq/00000-stylegan3-r-ffhq16-gpus8-batch2048/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq64.zip \
        --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 95000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00001-stylegan3-r-ffhq32-gpus8-batch2048/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq128.zip \
        --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 57000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00002-stylegan3-t-ffhq64-gpus8-batch256/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq256.zip \
        --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 11000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00003-stylegan3-t-ffhq128-gpus8-batch256/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq512.zip \
        --gpus=8 --batch=128 --mirror=1 --snap 10 --batch-gpu 8 --kimg 4000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00004-stylegan3-t-ffhq256-gpus8-batch256/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq1024.zip \
        --gpus=8 --batch=128 --mirror=1 --snap 10 --batch-gpu 8 --kimg 4000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00005-stylegan3-t-ffhq512-gpus8-batch128/best_model.pkl
```

You can change `--gpus` and `--batch-gpu` arguments according to your environment. 

## Pretrained Model
We provide the following pretrained checkpoints:

|Dataset| Res | FID | PATH
 :---  |  ---:  |  ---:  | :---
ImageNet| 256<sup>2</sup>  |2.14|  <sub>`https://zenodo.org/record/10947091/files/stylesan-xl_imagenet256.pkl`</sub><br>
CIFAR10 | 32<sup>2</sup>   |1.36|  <sub>`https://zenodo.org/record/10947091/files/stylesan-xl_cifar10.pkl`</sub><br>
FFHQ    | 256<sup>2</sup>  |1.68|  <sub>`https://zenodo.org/record/10947091/files/stylesan-xl_ffhq256.pkl`</sub><br>
FFHQ    | 512<sup>2</sup>  |1.77|  <sub>`https://zenodo.org/record/10947091/files/stylesan-xl_ffhq512.pkl`</sub><br>
FFHQ    | 1024<sup>2</sup> |1.61|  <sub>`https://zenodo.org/record/10947091/files/stylesan-xl_ffhq1024.pkl`</sub><br>

To load the checkpoints, use PyTorch 2.1.1 (or later) as described in [Requirements](#requirements)

## Generating Samples
To generate a conditional sample sheet for ImageNet, run
```
python gen_class_samplesheet.py --outdir=sample_sheets --trunc=0.7 \
  --samples-per-class 6 --classes 95,207,449,713,927,992 --grid-width 12 \
  --network=<path_to_checkpoint>
```

To generate samples for FFHQ, run
```
python gen_images.py --outdir=out --seeds=0-1 --batch-sz 1 \
  --network=<path_to_checkpoint>
```

## Quality Metrics
You need to preprocess a dataset in advance, following Data Preparation.
To calculate metrics for a specific network snapshot, run
```
python calc_metrics.py --metrics=fid50k_full --network=<path_to_checkpoint>
python calc_metrics.py --metrics=is50k --network=<path_to_checkpoint>
```

## Citation
[1] Takida, Y., Imaizumi, M., Shibuya, T., Lai, C., Uesaka, T., Murata, N. and Mitsufuji, Y.,
"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer,"
ICLR 2024.
```
@inproceedings{takida2024san,
        title={{SAN}: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer},
        author={Takida, Yuhta and Imaizumi, Masaaki and Shibuya, Takashi and Lai, Chieh-Hsin and Uesaka, Toshimitsu and Murata, Naoki and Mitsufuji, Yuki},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=eiF7TU1E8E}
}
```