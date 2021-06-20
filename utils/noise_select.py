import numpy as np
import cv2

prt=np.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_1/val/transport_0.npy')
prt=np.reshape(prt,(800,800,9))
print(prt[:,:,0:1].max(),prt[:,:,0:1].min())
cv2.imwrite('/home/jcn/桌面/Nerf/nerf_my/utils/transports/prt_0.png',np.asarray(np.clip(prt[:,:,0:1]*255,0,255),dtype=np.uint8))

threshold=0.25
noise=np.asarray(prt[:,:,0]<threshold,np.float)
mask=cv2.imread('/home/jcn/桌面/Nerf/nerf_my/data/nerf_synthetic/lego/test/r_0_mask.png',flags=2)
mask=mask/255.0

noise=noise*mask
noise=np.asarray(np.clip(noise[:,:,None]*255,0,255),dtype=np.uint8)
cv2.imwrite('/home/jcn/桌面/Nerf/nerf_my/utils/transports/noise.png',noise)
cv2.imshow('noise',noise)
cv2.waitKey(0)


