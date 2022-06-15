import os
import shutil
import glob
dataset_dir='/home/xsy/datasets/cars/cars_train/'
allimg_dir=sorted(glob.glob(dataset_dir+'/*.png'))
num=len(allimg_dir)
os.chdir(dataset_dir)
try:
    os.makedirs( os.path.join(dataset_dir,'/src'))
except OSError:
    pass

try:
    os.makedirs(os.path.join(dataset_dir,'/trg'))
except OSError:
    pass
for i in range(0,num):
    if i%500==0:
        print(f'moving {i} image')
    img_path=allimg_dir[i]
    if i<num//2:
        des_path='./src'
    else:
        des_path='./trg'
    shutil.move(img_path,des_path)


print(f'successfully move {num+1} images!')