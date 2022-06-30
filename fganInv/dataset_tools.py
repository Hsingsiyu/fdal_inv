import os
import shutil
import glob
from PIL import Image
import numpy as np
# dataset_dir='/home/xsy/datasets/cat_jpg'
# allimg_dir=sorted(glob.glob(dataset_dir+'/*.webp'))
# num=len(allimg_dir)
# os.chdir(dataset_dir)
# os.makedirs(os.path.join(dataset_dir,'train/src'))
# os.makedirs(os.path.join(dataset_dir,'train/trg'))
#
# for i in range(0,num):
#     img_path=allimg_dir[i]
#     if i% 50000==0:
#         print(i)
#     # print(img_path)
#     # break
#     if i<(num//2):
#         des_path=os.path.join(dataset_dir,'train/src')
#     else:
#         des_path=os.path.join(dataset_dir,'train/trg')
#     shutil.move(img_path,des_path)
#
#
# print(f'successfully move {num+1} images!')
for filepath,dirname,filenames in os.walk('/home/xsy/datasets/cars'):
    for filename in filenames:
        img_pth=os.path.joint(filepath,filename)
        filename=os.path.splitext(filename)[0]
        img = np.asarray(Image.open(img_pth))
        crop = np.min(img.shape[:2])

        img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
              (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
        img = Image.fromarray(img, 'RGB')
        img = img.resize((512, 384), Image.ANTIALIAS)
        img.save(os.path.joint(filepath,filename+'.png'))
