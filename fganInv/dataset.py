import os
import pickle

import glob
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as trans

# import torch
# from torch.autograd import Variable

from PIL import Image, ImageDraw
import math
import cv2
import imgaug.augmenters as iaa
def brush_stroke_mask(img, color=(255,255,255)):
    # input :image,   code from: GPEN
    min_num_vertex = 5
    max_num_vertex = 8 #[8,10]
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 20
        mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img #Image.fromarray(img)
        # for _ in range(np.random.randint(1, 2)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)#[2*pi/5-2*pi/15]  4/15
        angle_max = mean_angle + np.random.uniform(0, angle_range)#[2*pi/5+2*pi/15] 8/15
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=color, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

class ImageDataset(data.Dataset):
    def __init__(self, dataset_args,train=True):
        self.root=dataset_args.data_root
        self.train=train   # train or val
        self.transform = trans.Compose([trans.ToTensor()])
        self.transform_t=self.transform
        if self.train:
            self.source_list=sorted(glob.glob(self.root+'/train/src/*.*'))
            self.target_list=sorted(glob.glob(self.root+'/train/trg/*.*'))
        else:
            self.source_list=sorted(glob.glob(self.root+'/test/*.*'))
            self.target_list=sorted(glob.glob(self.root+'/test/*.*'))
        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.size = dataset_args.size
        # self.cloud =iaa.CloudLayer(
        #         intensity_mean=(196, 255),
        #         intensity_freq_exponent=(-2.5, -2.0),
        #         intensity_coarse_scale=10,
        #         alpha_min=0,
        #         alpha_multiplier=(0.25, 0.75),
        #         alpha_size_px_max=(2, 8),
        #         alpha_freq_exponent=(-2.5, -2.0),
        #         sparsity=(0.8, 1.0),
        #         density_multiplier=(0.5, 1.0),
        #     )
        self.rain=iaa.RainLayer(
            density=(0.03, 0.14),
            density_uniformity=(0.8, 1.0),
            drop_size=(0.01, 0.02),
            drop_size_uniformity=(0.2, 0.5),
            angle=(-15, 15),
            speed=(0.05, 0.20),
            blur_sigma_fraction=(0.001, 0.001),
        )
        # self.snow = iaa.Snowflakes(flake_size=(0.2, 0.5), speed=(0.007, 0.02)) #only for cars
        self.gn=iaa.GaussianBlur(sigma=(0.0, 2))
        self.sp= iaa.SaltAndPepper(0.05)

    def __getitem__(self, index):
        item_s = self.transform(Image.open(self.source_list[index % len(self.source_list)]))
        img_t = Image.open(self.target_list[index % len(self.target_list)])
        temp_num=np.random.randint(4)
        if temp_num==0:
            img_t=brush_stroke_mask(img_t)
            # img_t_aug = self.snow(image=np.array(img_t))
            # img_t = Image.fromarray(np.uint8(img_t_aug))
        # elif temp_num==1:
        #     img_t_aug = self.cloud(image=np.array(img_t))
        #     img_t = Image.fromarray(np.uint8(img_t_aug))
        elif temp_num==1:
            img_t_aug = self.rain(image=np.array(img_t))
            img_t = Image.fromarray(np.uint8(img_t_aug))
        elif temp_num==2:
            img_t_aug = self.gn(image=np.array(img_t))
            img_t = Image.fromarray(np.uint8(img_t_aug))
        elif  temp_num==3:
            img_t_aug = self.sp(image=np.array(img_t))
            img_t = Image.fromarray(np.uint8(img_t_aug))

        item_t=self.transform_t(img_t)
        item_s=item_s*(self.max_val - self.min_val) + self.min_val
        item_t=item_t*(self.max_val - self.min_val) + self.min_val
        return {'x_s': item_s, 'x_t': item_t}

    def __len__(self):
        return max(len(self.source_list), len(self.target_list))

class geometryDataset(data.Dataset):
    def __init__(self, dataset_args,train=True):
        self.root=dataset_args.data_root
        self.train=train   # train or val
        self.transform = trans.Compose([trans.ToTensor()])
        if self.train:
            self.source_list=sorted(glob.glob(self.root+'/train/src/*.*'))
            self.target_list=sorted(glob.glob(self.root+'/train/trg/*.*'))
        else:
            self.source_list=sorted(glob.glob(self.root+'/test/*.*'))
            self.target_list=sorted(glob.glob(self.root+'/test/*.*'))
        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.size = dataset_args.size

        self.affine_transform=trans.Compose([trans.RandomAffine(degrees=(-30, 30),resample=3)])
        self.zoom_transform=trans.Compose([trans.CenterCrop([225,225]),trans.Resize([256,256])])
        self.pad_transform=trans.Compose([trans.Pad(padding=[30],fill=0),trans.Resize([256,256])])

    def __getitem__(self, index):
        item_s = self.transform(Image.open(self.source_list[index % len(self.source_list)]))
        img_t = Image.open(self.target_list[index % len(self.target_list)])
        choice=index%3
        if choice==1:
            img_t=self.affine_transform(img_t)
        elif choice==2:
            img_t=self.pad_transform(img_t)
        else:
            img_t=self.zoom_transform(img_t)
        item_t=self.transform(img_t)
        item_s=item_s*(self.max_val - self.min_val) + self.min_val
        item_t=item_t*(self.max_val - self.min_val) + self.min_val
        return {'x_s': item_s, 'x_t': item_t}

    def __len__(self):
        return max(len(self.source_list), len(self.target_list))

if __name__=='__main__':
    class Config:
        data_root ='/home/xsy/ffhq_256'
        size =256
        min_val = -1.0
        max_val = 1.0
        # target_type='stroke'
        # split=1000 #65000
    datasets_args = Config()
    batch_size=32
    train_dataset = ImageDataset(datasets_args, train=False)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(train_dataloader)
    data = data_iter.next()
    import torchvision.utils as tvutils
    import torch
    with torch.no_grad():
        x=data['x_t']
        # writer.add_image("train",x_train,global_step=E_iterations)
    # tvutils.save_image(tensor=torch.cat([data['x_t'],data['x_s']], dim=0), fp='test.jpg', nrow=batch_size, normalize=True,
    #                    scale_each=True)
    tvutils.save_image(tensor=data['x_t'], fp='test.png', nrow=4, normalize=True,
                       scale_each=True)