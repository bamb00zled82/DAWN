# DAWN: Robust Neural Blind Deconvolution via Alternating Wiener Guidance

## Abstract

Blind image deconvolution (BID) is an ill-posed inverse problem that aims to jointly recover a sharp image and blur kernel from a single degraded observation. Under nontrivial and high noise, which is common in photography and scientific imaging such as microscopy, BID remains particularly challenging, as existing methods suffer from noise overfitting and inaccurate kernel estimation. Existing training-data-free learning methods for BID, such as those based on the deep image prior (DIP), often utilize external guidance by including model-based estimates as additional loss terms, which underexploits the information these estimates carry especially under nontrivial noise. To address these challenges, we propose DAWN, a training-data-free DIP framework that alternates between injecting model-based estimates directly into the model input and refining this imperfect estimate with the network's own prior. To provide more reliable kernel estimates for these injections, we apply autoencoding regularization to both the image and kernel networks. On natural and scientific images under varying levels of Gaussian and Poisson noise, DAWN outperforms state-of-the-art methods in both image and kernel quality (achieving up to +4.5 dB over the next best baseline on the Levin dataset at $\sigma = 0.1$). Without any training images, DAWN remains highly competitive with a state-of-the-art diffusion model on FFHQ, despite the latter being trained on a massive number of in-distribution images.

## Requirements

Python 3.9 or later with the following packages:

```
torch >= 1.13
torchvision >= 0.14
numpy >= 1.23
scipy >= 1.9
scikit-image >= 0.19
opencv-python >= 4.6
Pillow >= 9.0
matplotlib >= 3.5
tqdm >= 4.64
```

A CUDA-capable GPU is recommended; the script falls back to CPU if none is available.

## Datasets

We evaluate DAWN on four datasets. Full construction details (blur generation, noise levels, and evaluation protocol) are specified in the paper.

- **Levin**: retrieved from the [SelfDeblur](https://github.com/csdwren/SelfDeblur) codebase.
- **Microscopy**: retrieved from the W-DIP codebase (the `micro_good` folder).
- **Köhler**: available from the official benchmark page: https://webdav.tuebingen.mpg.de/pixel/benchmark4camerashake/
- **FFHQ**: available from the official repository: https://github.com/NVlabs/ffhq-dataset

## Running DAWN

The entry point is `dawn.py`, which runs DAWN once on a single grayscale blurry image:

```
python dawn.py --input <path/to/blurry.png> --gt <path/to/sharp.png> --kernel_size 19 --sigma 0.1
```

Arguments:

- `--input` (required): path to the blurry input image. Images are loaded as grayscale.
- `--kernel_size` (required): the square blur-kernel canvas size (e.g., `19` for a 19×19 kernel).
- `--gt` (optional): path to the sharp ground-truth image. When provided, the final shift-aligned PSNR and SSIM are printed to the terminal.
- `--sigma` (optional, default `0`): standard deviation of additive white Gaussian noise applied to the input, for reproducing the noisy settings in the paper (e.g., `0.05` or `0.1`).

All other hyperparameters are fixed to the paper configuration and hardcoded at the top of `dawn.py`.

The deblurred image and the estimated kernel are saved to the `results/` folder, and a live PSNR readout is shown during optimization when a ground-truth image is provided.
