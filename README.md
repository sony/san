# Slicing Adversarial Network (SAN) [ICLR 2024]

This repository contains the official PyTorch implementation of **"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer"** (*[arXiv 2301.12811](https://arxiv.org/abs/2301.12811)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

### [[Project Page]](https://ytakida.github.io/san/)


## Requirements
This repository builds on the codebase of [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl). Please refer to the repository.

- 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
- CUDA toolkit 11.1 or later.
- GCC 7 or later compilers. The recommended GCC version depends on your CUDA version; see for example, CUDA 11.4 system requirements.
- If you run into problems when setting up the custom CUDA kernels, we refer to the [Troubleshooting docs](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary) of the original StyleGAN3 repo and the following issues: https://github.com/autonomousvision/stylegan_xl/issues/23.
- Windows user struggling installing the env might find https://github.com/autonomousvision/stylegan_xl/issues/10
  helpful.
- Use the following commands with Miniconda3 to create and activate your PG Python environment:
  - ```conda env create -f environment.yml```
  - ```conda activate sgxl```

## Data Preparation
You can download the ImageNet dataset [here](https://image-net.org/). To preprocess the dataset, run
```
python dataset_tool_for_imagenet.py --source=<path_to_imagenet>/ILSVRC --dest=./data/imagenet256.zip \
  --resolution=256x256 --transform=center-crop
```

## Training
We trained our StyleSAN-XL model as follows:
```
python train.py --outdir=./training-runs/imagenet --cfg=stylegan3-t --data=./data/imagenet256.zip \
  --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 11000 --syn_layers 11 \
  --superres --cls_weight 8.0 --up_factor 2 --head_layers 7 \
  --path_stem https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet128.pkl \
  --cond True
```

This repository builds on the codebase of [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl). Please refer to the repository for more details.

## Pretrained Model
We provide the following pretrained checkpoint:

|Dataset| Res | FID | PATH
 :---  |  ---:  |  ---:  | :---
ImageNet| 256<sup>2</sup>  |2.14|  <sub>`https://zenodo.org/record/8140976/files/stylesan-xl_imagenet256.pkl`</sub><br>

## Generating Samples
To generate a conditional sample sheet, run
```
python gen_class_samplesheet.py --outdir=sample_sheets --trunc=0.7 \
  --samples-per-class 6 --classes 95,207,449,713,927,992 --grid-width 12 \
  --network=<path_to_checkpoint>
```

In our paper [[1](#citation)], we compared our trained checkpoint with the pretrained StyleGAN-XL [checkpoint](https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl).

## Quality Metrics
You need to preprocess the ImageNet dataset in advance, following [Data Preparation](#data-preparation).
To calculate metrics for a specific network snapshot, run
```
python calc_metrics.py --metrics=fid50k_full --network=<path_to_checkpoint>
python calc_metrics.py --metrics=is50k --network=<path_to_checkpoint>
```

## Citation
[1] Takida, Y., Imaizumi, M., Shibuya, T., Lai, C., Uesaka, T., Murata, N. and Mitsufuji, Y.,
"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer,"
Preprint.
```
@ARTICLE{takida2023san,
    author={Takida, Yuhta and Imaizumi, Masaaki and Shibuya, Takashi and Lai, Chieh-Hsin and Uesaka, Toshimitsu and Murata, Naoki and Mitsufuji, Yuki},
    title={{SAN}: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer},
    journal={Computing Research Repository},
    volume={arXiv:2301.12811},
    year={2023},
    url={https://arxiv.org/abs/2301.12811},
    }
```