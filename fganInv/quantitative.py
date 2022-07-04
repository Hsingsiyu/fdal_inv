import os
import glob

from PIL import Image
import torch
from torchvision import datasets,transforms
from swd import swd

import lpips
import piq
# from piq import ssim,psnr
# PSNR
from piqa import PSNR
# SSIM
from piqa import SSIM

import torch
import numpy as np
import cv2



if __name__=='__main__':
    torch.manual_seed(123)
    #fix seed
    path1='/home/xsy/idinvert_pytorch-mycode/results/0427_quan_val/src'# referece
    path2='' # reconstructed



    batch_size=900
    data_transforms=transforms.Compose([
        transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img1=datasets.ImageFolder(path1,transform=None)
    imgLoader_1=torch.utils.data.DataLoader(img1,batch_size=None)

    img2=datasets.ImageFolder(path2,transform=None)
    imgLoader_2=torch.utils.data.Dataloader(img2,batch_size=batch_size)


    x,labels=next(iter(imgLoader_1))#[BN,3,W,H]
    y,labels=next(iter(imgLoader_2))#[BN,3,W,H]

    x=x.cuda()
    y=y.cuda()


    #[0,1]
    ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(x, y, data_range=255.)
    # ms_ssim_loss = piq.MultiScaleSSIMLoss(data_range=1., reduction='none')(x, y)
    print(f"MS-SSIM index: {ms_ssim_index.item():0.4f}")

    psnr_index = piq.psnr(x, y, data_range=255., reduction='mean')
    print(f"PSNR index: {psnr_index.item():0.4f}")

    ssim_index = piq.ssim(x, y, data_range=255.)
    print(f"SSIM index: {ssim_index.item():0.4f}")

    fid_metric=piq.FID()
    x_feats = fid_metric.compute_feats(x)
    y_feats = fid_metric.compute_feats(y)
    fid: torch.Tensor = piq.FID()(x_feats, x_feats)
    print(f"FID: {fid:0.4f}")

    kid: torch.Tensor = piq.KID()(x, y)
    print(f"KID: {kid:0.4f}")

    swd_out=swd(x,y,device="cuda")#[0,1]


    MSE_out=torch.mean((x-y)**2) #[0,255]


    # # # FID,SWD
    # # # MSE,LPIPS,
    # # # SSIM,PSNR

    #[0,1]
    with torch.no_grad():
        loss_fn_alex=lpips.LPIPS(net='alex').cuda()
        d = loss_fn_alex.forward(x, y,normalize=True)


#
#     d_alex = loss_fn_alex((origin_img), (baseline_img))
#     d_vgg=loss_fn_vgg((origin_img), (baseline_img))
# print(f"LPIPS baseline: {d_alex.mean().item():.3f} {d_vgg.mean().item():.3f}")
#
# with torch.no_grad():
#     d_alex = loss_fn_alex((origin_img), (our_img))
#     d_vgg=loss_fn_vgg((origin_img), (our_img))
# print(f"LPIPS our: {d_alex.mean().item():.3f} {d_vgg.mean().item():.3f}")

