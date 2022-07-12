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
# from piqa import PSNR
# SSIM
# from piqa import SSIM

import torch
import numpy as np
import cv2
from torchvision import transforms as trans


class CustomDataset(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, path,trans=None):
        self.img_list = sorted(glob.glob(path + '/*.*'))
        self.trans=trans
        # self.transform = trans.Compose([trans.ToTensor()])
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        img=Image.open(self.img_list[index])
        if  self.trans is not None:
            img=self.trans(img)
        return np.array(img)

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.img_list)



if __name__=='__main__':
    torch.manual_seed(123)
    #fix seed
    path1='/home/xsy/datasets/evaluationt_img/src'# referece
    # path2='/home/xsy/invganV2/fganInv/results/inversion_ours/celebA1500_styleganinv_ffhq256_inpainting_styleganinv_encoder_epoch_050/inverted_img' # reconstructed
    path2='/home/xsy/SOTAgan_inversion/hyperstyle/result-src/inference_results/4'
    batch_size=1000
    # data_transforms=transforms.Compose([
    #     # transforms.ToTensor(),
    #     transforms.Resize(256),
    # # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    img1=CustomDataset(path1)
    imgLoader_1=torch.utils.data.DataLoader(img1,batch_size=batch_size,shuffle=False)

    img2=CustomDataset(path2,trans=None)
    imgLoader_2=torch.utils.data.DataLoader(img2,batch_size=batch_size,shuffle=False)

    x=next(iter(imgLoader_1))#[BN,3,W,H]
    y=next(iter(imgLoader_2))#[BN,3,W,H]

    x=torch.tensor(x).float().cuda()
    y=torch.tensor(y).float().cuda()
    x=x.permute(0,3,1,2)
    y=y.permute(0,3,1,2)
    #[0,1]
    with torch.no_grad():
        ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(x, y, data_range=255.)
        print(f"MS-SSIM index: {ms_ssim_index.item():0.4f}")

        psnr_index = piq.psnr(x, y, data_range=255., reduction='mean')
        print(f"PSNR index: {psnr_index.item():0.4f}")

        ssim_index = piq.ssim(x, y, data_range=255.)
        print(f"SSIM index: {ssim_index.item():0.4f}")

    x,y=x/255.0,y/255.0
    # item_s = item_s * (self.max_val - self.min_val) + self.min_val
    x=x*2-1
    y=y*2-1
    with torch.no_grad():
        swd_out=swd(x,y,device="cuda")#[0,1]
        MSE_out=torch.mean((x-y)**2) #[-1,1]
        # MSE_out=torch.mean((x-y).norm(2).pow(2))
        print(f"MSE index: {MSE_out.item():0.4f}")
    print(f"SWD: {swd_out:0.4f}")
    #[0,1]
    with torch.no_grad():
        loss_fn_alex=lpips.LPIPS(net='alex').cuda()
        # d = loss_fn_alex.forward(x, y,normalize=True)
        d = loss_fn_alex.forward(x, y,normalize=False)

    print(f"LPIPS: {d.mean().item():0.4f}")

