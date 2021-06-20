import cv2
import os
import numpy as np

root='/home/jcn/桌面/Nerf/nerf_my/data/nerf_synthetic/lego/test'
for i in range(200):
    image_path=os.path.join(root,f'r_{i}_normal_0001.png')
    mask=cv2.imread(image_path,flags=2)
    mask=np.asarray((mask>0),dtype=float)
    mask=np.asarray(255*mask[:,:,None],dtype=np.uint8)
    mask_path=os.path.join(root,f'r_{i}_mask.png')
    cv2.imwrite(mask_path,mask)
