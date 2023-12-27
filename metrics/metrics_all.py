import cv2
import numpy as np
from os import path as osp
import os
import math
import warnings
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import bgr2ycbcr, scandir
from basicsr.metrics import calculate_niqe
from basicsr.utils import scandir
import lpips
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor
import glob
from tqdm import tqdm
import torch
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader
from basicsr.data import build_dataset
from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3

def calculate_psnr_ssim(gt_path, restored_path, test_y_channel = False, crop_border = 0, suffix = '', correct_mean_var = False, show_details =False):
    """
    Calculate PSNR and SSIM for images.
    gt_path: Path to gt (Ground-Truth)
    restored_path: Path to restored images
    test_y_channel: If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.
    crop_border: Crop border for each side
    suffix: Suffix for restored images
    """
    print("Calculate PSNR and SSIM for images")
    psnr_all = []
    ssim_all = []
    img_list_gt = sorted(list(scandir(gt_path, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(restored_path, recursive=True, full_path=True)))

    if test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in tqdm(enumerate(img_list_gt)):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if suffix == '':
            img_path_restored = img_list_restored[i]
        else:
            img_path_restored = osp.join(restored_path, basename + suffix + ext)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # img_restored = cv2.imread(img_path_restored, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_restored
        if correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=crop_border, input_order='HWC')
        if show_details:
            print(f'{basename + suffix + ext:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    Average_psnr = sum(psnr_all) / len(psnr_all)
    Average_ssim = sum(ssim_all) / len(ssim_all)
    print(f'PSNR: {Average_psnr:.6f} dB, SSIM: {Average_ssim:.6f}')
    return Average_psnr, Average_ssim


def calculate_lpips(gt_path, restored_path, suffix = '', show_details =False):
    """
    Calculate LPIPS for images.
    gt_path: Path to gt (Ground-Truth)
    restored_path: Path to restored images
    suffix: Suffix for restored images
    """
    print("Calculate LPIPS for images")
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(gt_path, '*')))
    img_list_restored = sorted(list(scandir(restored_path, recursive=True, full_path=True)))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in tqdm(enumerate(img_list)):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if suffix == '':
            img_path_restored = img_list_restored[i]
        else:
            img_path_restored = osp.join(restored_path, basename + suffix + ext)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.  
        # img_restored = cv2.imread(img_path_restored, cv2.IMREAD_COLOR).astype(np.float32) / 255.  

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())
        lpips_val = lpips_val.cpu().item()
        if show_details:
            print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)
    Average_lpips = sum(lpips_all) / len(lpips_all)
    print(f'LPIPS: {Average_lpips:.6f}')
    return Average_lpips


def calculate_NIQE(restored_path, crop_border = 0, show_details =False):
    """
    Calculate NIQE for images.
    restored_path: Path to restored images
    crop_border: Crop border for each side
    """
    print("Calculate NIQE for images")
    niqe_all = []
    img_list = sorted(scandir(restored_path, recursive=True, full_path=True))

    for i, img_path in tqdm(enumerate(img_list)):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, crop_border, input_order='HWC', convert_to='y')
        if show_details:
            print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all.append(niqe_score)
    Average_niqe = sum(niqe_all) / len(niqe_all)
    print(f'NIQE: {Average_niqe:.6f}')
    return Average_niqe    

def load_image(img_path):
    image = cv2.imread(img_path, 0)  # only on gray images
    # resise
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))
    # image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    image = torch.from_numpy(image)
    return image


def load_image_torch(img_path):
    image = cv2.imread(img_path) / 255.
    image = image.astype(np.float32)
    image = img2tensor(image, bgr2rgb=True, float32=True)
    normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    image.unsqueeze_(0)
    image = (0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :])
    image = image.unsqueeze(1)
    image = F.interpolate(image, (128, 128), mode='bilinear', align_corners=False)
    return image


def calculate_fid_folder(restored_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fid_stats = ''
    batch_size = 64
    num_sample = 50000
    num_workers = 4
    backend = 'disk'

    # inception model
    inception = load_patched_inception_v3(device)

    # create dataset
    opt = {}
    opt['name'] = 'SingleImageDataset'
    opt['type'] = 'SingleImageDataset'
    opt['dataroot_lq'] = restored_path
    opt['io_backend'] = dict(type=backend)
    opt['mean'] = [0.5, 0.5, 0.5]
    opt['std'] = [0.5, 0.5, 0.5]
    dataset = build_dataset(opt)

    # create dataloader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        drop_last=False)
    num_sample = min(num_sample, len(dataset))
    total_batch = math.ceil(num_sample / batch_size)

    def data_generator(data_loader, total_batch):
        for idx, data in enumerate(data_loader):
            if idx >= total_batch:
                break
            else:
                yield data['lq']

    features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
    features = features.numpy()
    total_len = features.shape[0]
    features = features[:num_sample]
    print(f'Extracted {total_len} features, use the first {features.shape[0]} features to calculate stats.')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    # load the dataset stats
    stats = torch.load(fid_stats)
    real_mean = stats['mean']
    real_cov = stats['cov']

    # calculate FID metric
    fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
    print('fid:', fid)
    return fid


if __name__ == '__main__':
    gt_path = ""
    restored_path = ""
    show_details = False

    psnr, ssim = calculate_psnr_ssim(gt_path, restored_path, show_details=show_details)
    niqe = calculate_NIQE(restored_path, show_details=show_details)
    lpips = calculate_lpips(gt_path, restored_path, show_details=show_details)
    fid = calculate_fid_folder(restored_path)
    print(restored_path)
