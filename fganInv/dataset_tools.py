import os
import shutil
import glob
dataset_dir='/home/xsy/datasets/cat_jpg'
allimg_dir=sorted(glob.glob(dataset_dir+'/*.webp'))
num=len(allimg_dir)
os.chdir(dataset_dir)
os.makedirs(os.path.join(dataset_dir,'train/src'))
os.makedirs(os.path.join(dataset_dir,'train/trg'))

for i in range(0,num):
    img_path=allimg_dir[i]
    if i% 50000==0:
        print(i)
    # print(img_path)
    # break
    if i<(num//2):
        des_path=os.path.join(dataset_dir,'train/src')
    else:
        des_path=os.path.join(dataset_dir,'train/trg')
    shutil.move(img_path,des_path)


print(f'successfully move {num+1} images!')