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
    min_num_vertex = 8
    max_num_vertex = 8
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 20
        mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img #Image.fromarray(img)
        # for _ in range(np.random.randint(1, 2)):
        num_vertex = 5#np.random.randint(min_num_vertex, max_num_vertex)
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


# returns a list of lines
def get_noise(img, value=10):
    '''
    #生成噪声图像
    >>> 输入： img图像

        value= 大小控制雨滴的多少
    >>> 返回图像大小的模糊噪声图像
    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    return noise


def rain_blur(img,length=10, angle=0, w=5):
    '''将噪声加上运动模糊,模仿雨滴
    >>>输入
    noise：输入噪声图，shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle： 倾斜的角度，逆时针为正
    w:      雨滴大小
    >>>输出带模糊的噪声

    '''
    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    noise=get_noise(img)
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波
    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def alpha_rain(img, beta=0.8):
    # 输入雨滴噪声和图像
    # beta = 0.8   #results weight
    # 显示下雨效果

    # expand dimensin
    # 将二维雨噪声扩张为三维单通道
    # 并与图像合成在一起形成带有alpha通道的4通道图像
    rain=rain_blur(img)
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()  # 拷贝一个掩膜
    rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    return rain_result

class ImageDataset(data.Dataset):
    def __init__(self, dataset_args,train=True,paired=True):
        self.root=dataset_args.data_root
        self.target_type=dataset_args.target_type
        self.train=train   # train or val
        self.split=dataset_args.split
        self.paired=paired
        self.transform = trans.Compose([trans.ToTensor()])
        self.transform_t=self.transform
        if self.train:
            self.files_s=sorted(glob.glob(self.root+'/train/src/*.*'))
            self.files_t=sorted(glob.glob(self.root+'/train/trg/*.*'))
            self.source_list = self.collect_image(source=True)
            self.target_list = self.collect_image(source=False)
        else:
            self.source_list=sorted(glob.glob(self.root+'/test/*.*'))
            self.target_list=sorted(glob.glob(self.root+'/test/*.*'))

        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.size = dataset_args.size
        self.cloud =iaa.CloudLayer(
                intensity_mean=(196, 255),
                intensity_freq_exponent=(-2.5, -2.0),
                intensity_coarse_scale=10,
                alpha_min=0,
                alpha_multiplier=(0.25, 0.75),
                alpha_size_px_max=(2, 8),
                alpha_freq_exponent=(-2.5, -2.0),
                sparsity=(0.8, 1.0),
                density_multiplier=(0.5, 1.0),
            )
        self.rain=iaa.RainLayer(
            density=(0.03, 0.14),
            density_uniformity=(0.8, 1.0),
            drop_size=(0.01, 0.02),
            drop_size_uniformity=(0.2, 0.5),
            angle=(-15, 15),
            speed=(0.05, 0.20),
            blur_sigma_fraction=(0.001, 0.001),
        )
        self.gn=iaa.GaussianBlur(sigma=(0.0, 1))
        self.sp= iaa.SaltAndPepper(0.03, per_channel=True)
        # meshtemp=np.load('mesh_weight.npy')#[num,256,256]->[256,256,num]
        # self.mesh=meshtemp.transpose(1,2,0)

        # self.cloud=iaa.RainLayer(
        #     density=(0.03, 0.14),
        #     density_uniformity=(0.8, 1.0),
        #     drop_size=(0.01, 0.02),
        #     drop_size_uniformity=(0.2, 0.5),
        #     angle=(-15, 15),
        #     blur_sigma_fraction=(0.001, 0.001),
        #     speed=(0.04, 0.20),
        # )
    def collect_image(self,source=True):
        if self.train:
            if source:
                image_path_list=self.files_s[:int(self.split)]
            else: #train target
                image_path_list = self.files_t[:int(self.split)]
        return image_path_list

    def __getitem__(self, index):
        item_s = self.transform(Image.open(self.source_list[index % len(self.source_list)]))
        img_t = Image.open(self.target_list[index % len(self.target_list)])

        # TODO: add more type perturb  real world ESRGAN
        temp_num=index%5
        if temp_num==0:
            img_t=brush_stroke_mask(img_t)
        elif temp_num==1:
            img_t_aug = self.cloud(image=np.array(img_t))
            img_t = Image.fromarray(np.uint8(img_t_aug))
        elif temp_num==2:
            img_t_aug = self.rain(image=np.array(img_t))
            img_t = Image.fromarray(np.uint8(img_t_aug))
        elif temp_num==3:
            img_t_aug = self.gn(image=np.array(img_t))
            img_t = Image.fromarray(np.uint8(img_t_aug))
        elif  temp_num==4:
            img_t_aug = self.sp(image=np.array(img_t))
            img_t = Image.fromarray(np.uint8(img_t_aug))
            # img_t_aug=self.mesh[:,:,np.random.randint(low=8,size=1)]*np.array(img_t)
            # img_t = Image.fromarray(np.uint8(img_t_aug))
        # else:
            # img_t=np.array(img_t)
            # img_t_aug=self.rain(img_t)
            # img_t=Image.fromarray(np.uint8(img_t_aug))
        item_t=self.transform_t(img_t)
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
        target_type='stroke'
        split=1000 #65000
    datasets_args = Config()
    batch_size=32
    train_dataset = ImageDataset(datasets_args, train=False, paired=True)
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