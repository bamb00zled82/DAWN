"""
Shared evaluation utilities for blind deconvolution experiments.
All baseline eval scripts import from here.

Usage:
  from eval_utils import (load_image, compute_mnc, compute_ssim_chw,
                          compute_dice_shifted, parse_image_info,
                          read_kernel_size, load_gt_kernel,
                          save_kernel_png, save_image_png, get_group, get_kid,
                          cleanup, generate_summary)
"""

import os, gc, glob, json
import numpy as np
import torch
from collections import defaultdict
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import filters as sk_filters
from scipy.signal import correlate2d
from scipy import interpolate as sp_interpolate
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Image I/O
# ============================================================
def load_image(path, grayscale=False):
    img = imread(path)
    if grayscale:
        if img.ndim == 3: img = img.mean(axis=-1)
        return (img.astype(np.float32) / 255.0)[None, :, :]  # (1,H,W)
    else:
        if img.ndim == 2: img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] == 4: img = img[:,:,:3]
        return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3,H,W)

def save_kernel_png(kernel_np, path):
    k = kernel_np.squeeze()
    k = k / (k.max() + 1e-12)
    imsave(path, (k * 255).astype(np.uint8))

def save_image_png(img_chw, path):
    if img_chw.shape[0] == 1:
        img = np.clip(img_chw[0], 0, 1)
    else:
        img = np.clip(img_chw.transpose(1, 2, 0), 0, 1)
    imsave(path, (img * 255).astype(np.uint8))


# ============================================================
# Dataset helpers
# ============================================================
def parse_image_info(imgname, dataset):
    """Returns (gt_img_base, kernel_id, group_id)."""
    if dataset in ('levin', 'micro_bad', 'micro_good'):
        parts = imgname.split('_')
        return parts[0], parts[1], parts[0]
    elif dataset == 'kohler':
        # e.g. 'kohler_sharp_1_kernel_08_blurred'
        kernel_id = None
        for i in range(1, 13):  # kernel_01 through kernel_12
            k = f'kernel_{i:02d}'
            if k in imgname: kernel_id = k; break
        gt_base = imgname.split('_kernel_')[0]      # 'kohler_sharp_1'
        return gt_base, kernel_id, gt_base
    else:  # lai, ffhq
        kernel_id = None
        for i in range(1, 9):  # kernel_01 through kernel_08
            k = f'kernel_{i:02d}'
            if k in imgname: kernel_id = k; break
        return imgname.split('_kernel_')[0], kernel_id, imgname.split('_')[0]

def read_kernel_size(kernel_id, gt_kernel_path):
    for ext in ['.png','.npy']:
        kpath = os.path.join(gt_kernel_path, f'{kernel_id}{ext}')
        if os.path.exists(kpath):
            if ext == '.npy': k = np.load(kpath).squeeze()
            else:
                k = imread(kpath)
                if k.ndim == 3: k = k[:,:,0]
            return list(k.shape)
    raise ValueError(f"GT kernel not found for {kernel_id} in {gt_kernel_path}")

def load_gt_kernel(kernel_id, gt_kernel_path):
    for ext in ['.npy','.png']:
        p = os.path.join(gt_kernel_path, f'{kernel_id}{ext}')
        if os.path.exists(p):
            if ext == '.npy': kern = np.load(p).squeeze()
            else:
                kern = imread(p).astype(np.float32)
                if kern.ndim == 3: kern = kern.mean(axis=-1)
            return kern / (kern.sum() + 1e-12)
    return None

def get_group(imgname):
    return imgname.split('_')[0]

def get_kid(imgname, dataset='lai'):
    if dataset in ('levin', 'micro_bad', 'micro_good'):
        return imgname.split('_')[1]
    if dataset == 'kohler':
        for i in range(1, 13):
            k = f'kernel_{i:02d}'
            if k in imgname: return k
        return 'unknown'
    for k in ['kernel_01','kernel_02','kernel_03','kernel_04']:
        if k in imgname: return k
    return 'unknown'

def get_dataset_defaults(dataset):
    """Returns (data_path, gt_path, is_grayscale, results_prefix, ds_label)."""
    defaults = {
        'lai':       ('datasets/lai/tiny/', 'datasets/lai/ground_truth/',
                      False, 'results_tiny', 'Lai'),
        'levin':     ('datasets/levin/',    'datasets/levin/gt/',
                      True,  'results_levin', 'Levin'),
        'ffhq':      ('datasets/ffhq/blur/', 'datasets/ffhq/gt/',
                      False, 'results_ffhq', 'FFHQ'),
        'kohler':    ('datasets/kohler/blur/', 'datasets/kohler/gt/',
                      False, 'results_kohler', 'Kohler'),
        'micro_bad': ('datasets/micro_bad/blur/', 'datasets/micro_bad/gt/',
                      True,  'results_micro_bad', 'Micro Bad'),
        'micro_good':('datasets/micro_good/blur/', 'datasets/micro_good/gt/',
                      True,  'results_micro_good', 'Micro Good'),
    }
    return defaults[dataset]

def list_images(data_path, dataset):
    files = sorted(glob.glob(os.path.join(data_path, '*.png')))
    if dataset == 'levin':
        files = [f for f in files if '_img' in os.path.basename(f)]
    elif dataset in ('ffhq', 'kohler'):
        files = [f for f in files if '_blurred' in os.path.basename(f)]
    return files


# ============================================================
# Metrics
# ============================================================
def compute_mnc(k_est, k_gt, flip_safe=True):
    """Maximum Normalized Correlation between estimated and GT kernels.
    If flip_safe=True (default), tests both orientations and returns the max,
    handling the conv-vs-correlation ambiguity from F.conv2d."""
    k_est = k_est.squeeze().astype(np.float64)
    k_gt = k_gt.squeeze().astype(np.float64)
    n1, n2 = np.linalg.norm(k_est), np.linalg.norm(k_gt)
    if n1 < 1e-12 or n2 < 1e-12: return 0.0
    mnc = float(np.max(correlate2d(k_est/n1, k_gt/n2, mode='full')))
    if flip_safe:
        k_gt_flip = k_gt[::-1, ::-1].copy()
        mnc_flip = float(np.max(correlate2d(k_est/n1, k_gt_flip/n2, mode='full')))
        mnc = max(mnc, mnc_flip)
    return mnc

def compute_ssim_chw(gt_chw, est_chw):
    """SSIM for CHW arrays. Handles 1-channel and 3-channel."""
    if gt_chw.shape[0] == 1:
        return float(compare_ssim(gt_chw[0], est_chw[0], data_range=1.0))
    else:
        gt_hwc = gt_chw.transpose(1, 2, 0)
        est_hwc = est_chw.transpose(1, 2, 0)
        return float(compare_ssim(gt_hwc, est_hwc, data_range=1.0, channel_axis=2))

def compute_psnr_ssim_shifted(est_chw, gt_chw, maxshift=5, limit=15, kernel_size=None):
    """
    Shift-invariant PSNR and SSIM following Levin et al. (CVPR 2009)
    comp_upto_shift.m.  Standard evaluation for blind deconvolution.

    Blind deconvolution is shift-invariant: a shifted kernel paired with a
    correspondingly shifted image produces the same blurry observation.
    This finds the sub-pixel shift (quarter-pixel resolution) that minimizes
    MSE, then computes PSNR/SSIM on the aligned, border-cropped pair.

    Args:
        est_chw: estimated image, (C, H, W) in [0, 1]
        gt_chw:  ground truth,    (C, H, W) in [0, 1]
        maxshift: max shift in pixels (search range ±maxshift).
                  Overridden by kernel_size if provided.
        limit: border crop in pixels (removes boundary artifacts).
               Auto-raised to >= maxshift if needed.
        kernel_size: (kH, kW) tuple or int. If provided, sets
                     maxshift = max(10, kH // 2) to scale with kernel size.
    Returns:
        (psnr, ssim)
    """
    # Dynamic maxshift based on kernel size
    if kernel_size is not None:
        kH = kernel_size[0] if hasattr(kernel_size, '__len__') else kernel_size
        maxshift = max(10, kH // 2)
    limit = max(limit, maxshift)  # ensure limit >= maxshift for valid indexing
    est_chw = est_chw.astype(np.float64)
    gt_chw = gt_chw.astype(np.float64)

    # Shift search on channel-mean (single channel for grayscale)
    est_mean = est_chw.mean(axis=0) if est_chw.shape[0] > 1 else est_chw[0]
    gt_mean = gt_chw.mean(axis=0) if gt_chw.shape[0] > 1 else gt_chw[0]

    # Crop: GT tight, est wider to allow shift room
    I2 = gt_mean[limit:-limit, limit:-limit]
    I1 = est_mean[limit - maxshift:est_mean.shape[0] - limit + maxshift,
                  limit - maxshift:est_mean.shape[1] - limit + maxshift]
    I2_h, I2_w = I2.shape

    gx = np.arange(-maxshift, I2_h + maxshift, 1)
    gy = np.arange(-maxshift, I2_w + maxshift, 1)
    gx0 = np.arange(0, I2_h, 1)
    gy0 = np.arange(0, I2_w, 1)

    f = sp_interpolate.RectBivariateSpline(gx, gy, I1)

    # Exhaustive sub-pixel search (quarter-pixel, matching Levin)
    shifts = np.arange(-maxshift, maxshift + 0.25, 0.25)
    mse_ar = np.zeros((len(shifts), len(shifts)))
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            Inew = f(gx0 + shifts[i], gy0 + shifts[j])
            mse_ar[i, j] = np.sum((Inew - I2) ** 2)

    idx = np.unravel_index(np.argmin(mse_ar), mse_ar.shape)
    best_sx, best_sy = shifts[idx[0]], shifts[idx[1]]

    # Apply optimal shift to each channel
    C = est_chw.shape[0]
    shifted_chs, gt_chs = [], []
    for c in range(C):
        I2c = gt_chw[c, limit:-limit, limit:-limit]
        I1c = est_chw[c, limit - maxshift:est_chw.shape[1] - limit + maxshift,
                         limit - maxshift:est_chw.shape[2] - limit + maxshift]
        fc = sp_interpolate.RectBivariateSpline(gx, gy, I1c)
        shifted_chs.append(fc(gx0 + best_sx, gy0 + best_sy))
        gt_chs.append(I2c)

    est_s = np.stack(shifted_chs, axis=0)
    gt_s = np.stack(gt_chs, axis=0)

    mse_val = np.mean((est_s - gt_s) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse_val) if mse_val > 0 else 100.0
    ssim = compute_ssim_chw(gt_s, est_s)

    return float(psnr), float(ssim)

def custom_otsu_seg(img, excl=40):
    """Otsu segmentation with bottom-percentile exclusion (from W-DIP)."""
    if torch.is_tensor(img): img = img.detach().cpu().numpy()
    img_temp = img.copy().flatten()
    thr = np.percentile(img_temp, excl)
    ind = np.argwhere(img_temp < thr)
    img_temp_new = np.delete(img_temp, ind)
    val = sk_filters.threshold_otsu(img_temp_new)
    return img > val

def compute_dice_shifted(img_est, img_gt, maxshift=10, limit=25):
    """
    Shift-aligned PSNR, SSIM, DICE evaluation (from W-DIP evaluation.py).
    Inputs: 2D arrays (H,W) in [0,1].
    Returns: (psnr, ssim, dice)
    """
    shifts = np.arange(-maxshift, maxshift + 0.25, 0.25)
    I2 = img_gt[limit:img_gt.shape[0]-limit, limit:img_gt.shape[1]-limit]
    I1 = img_est[limit-maxshift:img_est.shape[0]-limit+maxshift,
                 limit-maxshift:img_est.shape[1]-limit+maxshift]
    I2_x, I2_y = I2.shape
    gx = np.arange(-maxshift, I2_x + maxshift, 1)
    gy = np.arange(-maxshift, I2_y + maxshift, 1)
    gx0 = np.arange(0, I2_x, 1)
    gy0 = np.arange(0, I2_y, 1)
    f = sp_interpolate.RectBivariateSpline(gx, gy, I1)
    mse_ar = np.zeros((len(shifts), len(shifts)))
    for i in range(len(shifts)):
        for j in range(len(shifts)):
            Inew = f(gx0 + shifts[i], gy0 + shifts[j])
            mse_ar[i, j] = np.mean(np.sqrt((Inew - I2)**2))
    index = np.unravel_index(np.argmin(mse_ar), mse_ar.shape)
    I1_shift = f(gx0 + shifts[index[0]], gy0 + shifts[index[1]])
    # PSNR
    mse_val = np.mean((I1_shift - I2)**2)
    psnr = 10 * np.log10(1.0 / mse_val) if mse_val > 0 else 100.0
    # SSIM
    ssim = float(compare_ssim(I2, I1_shift, data_range=1.0))
    # DICE
    seg_est = custom_otsu_seg(I1_shift)
    seg_gt = custom_otsu_seg(I2)
    dice = float(np.sum(seg_est[seg_gt == 1]) * 2.0 /
                 (np.sum(seg_est) + np.sum(seg_gt) + 1e-12))
    return psnr, ssim, dice


# ============================================================
# Utilities
# ============================================================
def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ============================================================
# Summary generation (reads per-image JSONs, writes summary)
# ============================================================
def generate_summary(save_path, per_image_dir, dataset, variant_name, sigma):
    """Read all per-image JSONs and generate terminal summary, results.json, summary.png."""
    is_micro = dataset in ('micro_bad', 'micro_good')
    group_label = 'Image' if dataset in ('levin', 'micro_bad', 'micro_good') else 'Category'
    ds_label = {'lai': 'Lai', 'levin': 'Levin', 'micro_bad': 'Micro Bad',
                'micro_good': 'Micro Good', 'kohler': 'Kohler'}.get(dataset, dataset)

    all_results = {}
    for jf in sorted(glob.glob(os.path.join(per_image_dir, '*.json'))):
        imgname = os.path.splitext(os.path.basename(jf))[0]
        with open(jf) as fin:
            all_results[imgname] = json.load(fin)

    if not all_results:
        print("No results found."); return

    # Averages
    metrics = ['final_psnr','best_psnr','final_ssim','best_ssim','final_mnc','best_mnc']
    if is_micro: metrics += ['final_dice', 'best_dice']
    avgs = {m: float(np.mean([r.get(m, 0) for r in all_results.values()])) for m in metrics}

    # Terminal summary
    print(f"\n{'='*120}")
    print(f"  {variant_name} | {ds_label} | sigma={sigma} | {len(all_results)} images")
    print(f"{'='*120}")
    hdr = f"  {'Image':<40} {'FinalP':>7} {'BestP':>7} {'FinalS':>7} {'BestS':>7} {'FinalM':>7} {'BestM':>7}"
    if is_micro: hdr += f" {'FinalD':>7} {'BestD':>7}"
    print(hdr)

    for img in sorted(all_results.keys()):
        r = all_results[img]
        line = (f"  {img:<40} {r['final_psnr']:>7.2f} {r['best_psnr']:>7.2f} "
                f"{r['final_ssim']:>7.4f} {r['best_ssim']:>7.4f} "
                f"{r['final_mnc']:>7.4f} {r['best_mnc']:>7.4f}")
        if is_micro:
            line += f" {r.get('final_dice',0):>7.4f} {r.get('best_dice',0):>7.4f}"
        print(line)

    avg_line = (f"  {'AVERAGE':<40} {avgs['final_psnr']:>7.2f} {avgs['best_psnr']:>7.2f} "
                f"{avgs['final_ssim']:>7.4f} {avgs['best_ssim']:>7.4f} "
                f"{avgs['final_mnc']:>7.4f} {avgs['best_mnc']:>7.4f}")
    if is_micro:
        avg_line += f" {avgs['final_dice']:>7.4f} {avgs['best_dice']:>7.4f}"
    print(avg_line)

    # Per-group and per-kernel breakdown
    cat = defaultdict(list); kern = defaultdict(list)
    for img, r in all_results.items():
        cat[get_group(img)].append(r)
        kern[get_kid(img, dataset)].append(r)

    print(f"\n  Per {group_label}:")
    for g in sorted(cat.keys()):
        rs = cat[g]
        line = (f"    {g:<12}: PSNR={np.mean([r['final_psnr'] for r in rs]):.2f} "
                f"SSIM={np.mean([r['final_ssim'] for r in rs]):.4f} "
                f"MNC={np.mean([r['final_mnc'] for r in rs]):.4f}")
        if is_micro:
            line += f" DICE={np.mean([r.get('final_dice',0) for r in rs]):.4f}"
        print(line + f" (n={len(rs)})")

    print(f"\n  Per Kernel:")
    for ki in sorted(kern.keys()):
        rs = kern[ki]
        line = (f"    {ki:<12}: PSNR={np.mean([r['final_psnr'] for r in rs]):.2f} "
                f"SSIM={np.mean([r['final_ssim'] for r in rs]):.4f} "
                f"MNC={np.mean([r['final_mnc'] for r in rs]):.4f}")
        if is_micro:
            line += f" DICE={np.mean([r.get('final_dice',0) for r in rs]):.4f}"
        print(line + f" (n={len(rs)})")
    print(f"{'='*120}")

    # Combined results JSON
    with open(os.path.join(save_path, 'results.json'), 'w') as fout:
        json.dump({'per_image': all_results, 'averages': avgs}, fout, indent=2)

    # Summary PNG
    cats = sorted(cat.keys()); kerns = sorted(kern.keys())
    n_rows = 4 if is_micro else 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 4.5 * n_rows))
    suptitle = (f'{variant_name} | {ds_label} | sigma={sigma} | {len(all_results)} images\n'
                f'Avg: PSNR f={avgs["final_psnr"]:.2f} b={avgs["best_psnr"]:.2f} | '
                f'SSIM f={avgs["final_ssim"]:.4f} b={avgs["best_ssim"]:.4f} | '
                f'MNC f={avgs["final_mnc"]:.4f} b={avgs["best_mnc"]:.4f}')
    if is_micro:
        suptitle += f' | DICE f={avgs["final_dice"]:.4f} b={avgs["best_dice"]:.4f}'
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    def bar_row(row, metric_f, metric_b, ylabel, title_prefix, fmt='.2f'):
        w = 0.35
        ax = axes[row, 0]
        vals = [np.mean([r.get(metric_f, 0) for r in all_results.values()]),
                np.mean([r.get(metric_b, 0) for r in all_results.values()])]
        ax.bar([0, 0.5], vals, 0.35, color=['steelblue','salmon'], edgecolor='black')
        ax.set_xticks([0, 0.5]); ax.set_xticklabels(['Final', 'Best'])
        ax.set_ylabel(ylabel); ax.set_title(f'{title_prefix} Overall')
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[row, 1]; x = np.arange(len(kerns))
        ax.bar(x-w/2, [np.mean([r.get(metric_f, 0) for r in kern[k]]) for k in kerns], w,
               label='Final', color='steelblue', edgecolor='black')
        ax.bar(x+w/2, [np.mean([r.get(metric_b, 0) for r in kern[k]]) for k in kerns], w,
               label='Best', color='salmon', edgecolor='black')
        for i, k in enumerate(kerns):
            vf = np.mean([r.get(metric_f, 0) for r in kern[k]])
            vb = np.mean([r.get(metric_b, 0) for r in kern[k]])
            ax.text(i-w/2, vf + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.01,
                    f'{vf:{fmt}}', ha='center', fontsize=6)
            ax.text(i+w/2, vb + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.01,
                    f'{vb:{fmt}}', ha='center', fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(kerns, fontsize=7, rotation=30)
        ax.set_ylabel(ylabel); ax.set_title(f'{title_prefix} by Kernel')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

        ax = axes[row, 2]; x = np.arange(len(cats))
        ax.bar(x-w/2, [np.mean([r.get(metric_f, 0) for r in cat[c]]) for c in cats], w,
               label='Final', color='steelblue', edgecolor='black')
        ax.bar(x+w/2, [np.mean([r.get(metric_b, 0) for r in cat[c]]) for c in cats], w,
               label='Best', color='salmon', edgecolor='black')
        for i, c in enumerate(cats):
            vf = np.mean([r.get(metric_f, 0) for r in cat[c]])
            vb = np.mean([r.get(metric_b, 0) for r in cat[c]])
            ax.text(i-w/2, vf + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.01,
                    f'{vf:{fmt}}', ha='center', fontsize=6)
            ax.text(i+w/2, vb + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.01,
                    f'{vb:{fmt}}', ha='center', fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=7, rotation=15)
        ax.set_ylabel(ylabel); ax.set_title(f'{title_prefix} by {group_label}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    bar_row(0, 'final_psnr', 'best_psnr', 'PSNR (dB)', 'PSNR')
    bar_row(1, 'final_mnc', 'best_mnc', 'MNC', 'MNC', fmt='.3f')
    bar_row(2, 'final_ssim', 'best_ssim', 'SSIM', 'SSIM', fmt='.4f')
    if is_micro:
        bar_row(3, 'final_dice', 'best_dice', 'DICE', 'DICE', fmt='.4f')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSummary: {os.path.join(save_path, 'summary.png')}")
    print(f"Results: {os.path.join(save_path, 'results.json')}")
