import matplotlib.pyplot as plt
import numpy
import numpy as np
import cv2
num=4
temp_mat=np.zeros((num,256,256))
for i in range(num):
    mesh=cv2.imread(f'./mesh-template/0{i+1}.png',-1)
    gray=cv2.cvtColor(mesh,cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    template = gray.astype(np.float32) // 255
    print(template.sum())
    temp_mat[i]=template
    # cv2.imshow('img',mesh)
    # cv2.imshow('gray1',gray)
    # cv2.threshold(gray,140,255,0,cv2.THRESH_BINARY)#二值化函数
    # cv2.imshow('gray2',gray)


# np.save('mesh_weight.npy',temp_mat)
# # cv2.imshow('gray2',gray)
# # cv2.destoryAllWindows()
# # ret,img=cv2.threshold(mesh,127,255,cv2.THRESH_BINARY)
# # cv2.imshow("title",img)
# cv2.waitKey(0)