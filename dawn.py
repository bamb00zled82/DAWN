"""
DAWN: Robust Neural Blind Deconvolution via Alternating Wiener Guidance
------------------------------------------------------------------------
Single-image demo script (grayscale).

Runs DAWN once on a given blurry image and saves the deblurred result to
./results/. If a ground-truth sharp image is provided, the final
shift-aligned PSNR/SSIM is printed to the terminal.

Example:
    python dawn.py --input datasets/levin/im1_kernel1_img.png \
                   --gt datasets/levin/gt/im1.png \
                   --kernel_size 19 --sigma 0.1
"""
import argparse
import os
import numpy as np
import cv2
from numpy.fft import fft2, ifft2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from networks.skip import skip
from networks.fcn import fcn
from utils.common_utils import get_noise
from utils.eval_utils import (load_image, save_image_png, save_kernel_png,
                              compute_psnr_ssim_shifted)

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DAWN single-image blind deconvolution (grayscale)')
parser.add_argument('--input', type=str, required=True,
                    help='Path to the blurry input image (grayscale)')
parser.add_argument('--gt', type=str, default=None,
                    help='Optional path to the sharp ground-truth image (enables PSNR/SSIM)')
parser.add_argument('--kernel_size', type=int, required=True,
                    help='Blur kernel canvas size (square), e.g. 19 for a 19x19 kernel')
parser.add_argument('--sigma', type=float, default=0.0,
                    help='Std of additive white Gaussian noise applied to the input (default 0)')
opt = parser.parse_args()

# ============================================================
# Fixed hyperparameters (paper configuration)
# ============================================================
NUM_ITER = 4000      # total gradient steps
N = 2                # alternating phases per outer iteration
M = 200              # kernel-network (FCN) input dimension
LR_X = 1e-3          # image network learning rate
LR_K = 1e-5          # kernel network learning rate
LAMBDA_REG = 1.0     # weight of the z_x consistency term (image autoencoding)
LAMBDA_K = 2.0       # weight of the kernel autoencoding term
GAMMA_K = 0.5        # EMA rate for the kernel autoencoding target
ZHONG_EVERY = 200    # Wiener+NLM injection period (in outer iterations)
W3 = 0.05            # Wiener regularization weight
NLM_H = 20.0         # NLM denoising strength
REG_NOISE_STD = 0.01 # input perturbation std for the image network

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# ============================================================
# Wiener-guided injection (currently grayscale)
# ============================================================
def wrap_boundary(img, img_size):
    """Pad a 2D image to img_size with smooth wrap-around interpolation."""
    H, W = img.shape
    H_new, W_new = img_size
    if H_new == H and W_new == W:
        return img.copy()
    ret = np.zeros((H_new, W_new), dtype=img.dtype)
    ret[:H, :W] = img
    if W_new > W:
        t = np.linspace(0, 1, W_new - W + 2)[1:-1]
        ret[:H, W:] = img[:, -1:] * (1 - t[None, :]) + img[:, :1] * t[None, :]
    if H_new > H:
        t = np.linspace(0, 1, H_new - H + 2)[1:-1]
        ret[H:, :W] = img[-1:, :] * (1 - t[:, None]) + img[:1, :] * t[:, None]
    if H_new > H and W_new > W:
        th = np.linspace(0, 1, H_new - H + 2)[1:-1]
        tw = np.linspace(0, 1, W_new - W + 2)[1:-1]
        br, bl, tr, tl = img[-1, -1], img[-1, 0], img[0, -1], img[0, 0]
        for i, ah in enumerate(th):
            for j, aw in enumerate(tw):
                ret[H + i, W + j] = (br * (1 - ah) * (1 - aw) + bl * (1 - ah) * aw
                                     + tr * ah * (1 - aw) + tl * ah * aw)
    return ret


def zhong_one_iter(blurry_np, kernel_np, w3=W3, nlm_h=NLM_H):
    """One Wiener deconvolution + NLM denoising step (grayscale).

    Note: F.conv2d computes cross-correlation, so the learned kernel is the
    180-degree rotation of the convolution kernel; it is flipped back here
    before classical (convolution-convention) Wiener deconvolution.
    """
    H, W = blurry_np.shape
    kh, kw = kernel_np.shape
    kernel_flip = kernel_np[::-1, ::-1].copy()
    H_ext, W_ext = H + kh - 1, W + kw - 1
    blurry_ext = wrap_boundary(blurry_np, (H_ext, W_ext))
    k_pad = np.zeros((H_ext, W_ext), dtype=np.float64)
    k_pad[:kh, :kw] = kernel_flip
    k_pad = np.roll(k_pad, -(kh // 2), axis=0)
    k_pad = np.roll(k_pad, -(kw // 2), axis=1)
    K = fft2(k_pad)
    B = fft2(blurry_ext)
    L = (np.conj(K) * B) / (np.abs(K) ** 2 + w3)
    wiener = np.clip(np.real(ifft2(L))[:H, :W], 0, 1)
    w_uint8 = (wiener * 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(w_uint8, None, float(nlm_h), 7, 21)
    return denoised.astype(np.float64) / 255.0


# ============================================================
# Load data
# ============================================================
assert os.path.exists(opt.input), f"Input image not found: {opt.input}"
imgs = load_image(opt.input, grayscale=True)                 # (1,H,W) in [0,1]
y_clean = torch.from_numpy(imgs).unsqueeze(0).type(dtype)    # (1,1,H,W)
if opt.sigma > 0:
    y = torch.clamp(y_clean + opt.sigma * torch.randn_like(y_clean), 0, 1)
else:
    y = y_clean
blurry_np = y.cpu().numpy()[0, 0].astype(np.float64)
img_h, img_w = blurry_np.shape

gt_img = None
if opt.gt is not None:
    assert os.path.exists(opt.gt), f"GT image not found: {opt.gt}"
    gt_img = load_image(opt.gt, grayscale=True)              # (1,H,W)
    assert gt_img.shape[1:] == (img_h, img_w), \
        "GT and input image must have the same spatial size"

kH = kW = opt.kernel_size
padh, padw = kH - 1, kW - 1
pad_top = padh // 2; pad_bottom = padh - pad_top
pad_left = padw // 2; pad_right = padw - pad_left

# ============================================================
# Networks
# ============================================================
net = skip(1, 1,
           num_channels_down=[128] * 5, num_channels_up=[128] * 5,
           num_channels_skip=[16] * 5, upsample_mode='bilinear',
           need_sigmoid=True, need_bias=True,
           pad='reflection', act_fun='LeakyReLU').type(dtype)
net_input_kernel = get_noise(M, 'noise', (1, 1)).type(dtype).squeeze()
net_kernel = fcn(M, kH * kW).type(dtype)

# z_x: the physics-guided network input (initialized to the padded observation)
z_x = F.pad(y, (pad_left, pad_right, pad_top, pad_bottom),
            mode='reflect').detach().clone()
mse = nn.MSELoss().type(dtype)
with torch.no_grad():
    k_avg = net_kernel(net_input_kernel).detach().clone()

optimizer = torch.optim.Adam([
    {'params': net.parameters(), 'lr': LR_X},
    {'params': net_kernel.parameters(), 'lr': LR_K}])

if gt_img is not None:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# ============================================================
# Optimization
# ============================================================
K_iters = NUM_ITER // N
final_img = None
final_kernel = None

pbar = tqdm(range(K_iters), desc='DAWN')
for k in pbar:
    for n in range(N):
        optimizer.zero_grad()
        out_x = net(z_x + torch.randn_like(z_x) * REG_NOISE_STD)
        out_k = net_kernel(net_input_kernel)
        out_k_m = out_k.view(-1, 1, kH, kW)
        out_y = F.conv2d(out_x, out_k_m, padding=0)
        loss = (mse(out_y, y)
                + LAMBDA_REG * mse(out_x, z_x)
                + LAMBDA_K * mse(out_k, k_avg.detach()))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # Kernel autoencoding target (EMA)
        current_k = net_kernel(net_input_kernel).detach()
        k_avg = (1 - GAMMA_K) * k_avg + GAMMA_K * current_k

        # z_x update: periodic Wiener-guided injection, otherwise self-loop
        if k % ZHONG_EVERY == 0:
            kz = current_k.view(kH, kW).cpu().numpy()
            kz = np.clip(kz, 0, None).astype(np.float64)
            kz /= kz.sum() + 1e-12
            zi = zhong_one_iter(blurry_np, kz)
            zt = torch.from_numpy(zi.astype(np.float32)[None, :, :]).unsqueeze(0).type(dtype)
            z_x = F.pad(zt, (pad_left, pad_right, pad_top, pad_bottom),
                        mode='reflect').detach().clone()
        else:
            z_x = net(z_x).detach().clone()

        # Track current estimate
        out_np = out_x.cpu().numpy()[0]
        final_img = np.clip(out_np[:, padh // 2:padh // 2 + img_h,
                                   padw // 2:padw // 2 + img_w], 0, 1)
        final_kernel = out_k_m.cpu().numpy().squeeze()
        if gt_img is not None:
            curr_psnr = compare_psnr(gt_img, final_img, data_range=1.0)
            pbar.set_postfix_str(f"PSNR={curr_psnr:.2f}")
        else:
            pbar.set_postfix_str(f"loss={loss.item():.5f}")

# ============================================================
# Save outputs & report metrics
# ============================================================
os.makedirs('results', exist_ok=True)
stem = os.path.splitext(os.path.basename(opt.input))[0]
out_img_path = os.path.join('results', f'{stem}_deblurred.png')
out_ker_path = os.path.join('results', f'{stem}_kernel.png')
save_image_png(final_img, out_img_path)
save_kernel_png(final_kernel, out_ker_path)

print(f"\nDeblurred image saved to: {out_img_path}")
print(f"Estimated kernel saved to: {out_ker_path}")

if gt_img is not None:
    final_psnr, final_ssim = compute_psnr_ssim_shifted(
        final_img, gt_img, kernel_size=(kH, kW))
    print(f"\nFinal PSNR: {final_psnr:.2f} dB")
    print(f"Final SSIM: {final_ssim:.4f}")
