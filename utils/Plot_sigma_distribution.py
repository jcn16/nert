import numpy as np
import os
import matplotlib.pyplot as plot
import cv2

def Draw(points,z,h,w):
    plot.plot(z,points)
    plot.savefig(f'./figs/{h}_{w}.png')
    plot.cla()
    # plot.show()

def Note():
    root='/home/jcn/桌面/Nerf/nerf_my/data/nerf_synthetic/lego/val/r_0.png'
    image=cv2.imread(root)
    fill=np.zeros(shape=(100,3))
    image[350:450,400,:]=fill
    cv2.imwrite('/home/jcn/桌面/Nerf/nerf_my/utils/figs/r_0.png',image)

sigmas=np.load('/home/jcn/桌面/Nerf/nerf_my/utils/figs/sigmas.npy')
sigmas=np.reshape(sigmas,(800,800,192))
xyz=np.load('/home/jcn/桌面/Nerf/nerf_my/utils/figs/xyz.npy')
xyz=np.reshape(xyz,(800,800,192,3))

Note()
# for h in range(350,450):
#     w=400
#     Draw(sigmas[h,w,:],xyz[h,w,:,2],h,w)