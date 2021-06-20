import cv2
import numpy as np

prt=np.load('/media/jcn/新加卷/Nerf/Nerf/nerf_my/data/nerf_synthetic/lego/val/r_0_transport.npy',allow_pickle=True)

prt_1=np.asarray(np.clip(prt[:,:,0:1]*255.0,0,255),dtype=np.uint8)

cv2.imshow('image',prt_1)
cv2.waitKey(0)