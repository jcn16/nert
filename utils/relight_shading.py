from glob import glob
import numpy as np
import cv2
import os
import time
import sys
import torch
from torch import cuda
import sh_rot
import argparse
import random


def trim(img, mask, padding_x=5, padding_y=5):
    mask_ids = np.where(mask>0)
    y_max = min(max(mask_ids[0])+padding_y, img.shape[0])
    y_min = max(min(mask_ids[0])-padding_y, 0)
    x_max = min(max(mask_ids[1])+padding_x, img.shape[1])
    x_min = max(min(mask_ids[1])-padding_x, 0)
    if (y_max - y_min) % 2 == 1:
        y_max -= 1
    if (x_max - x_min) % 2 == 1:
        x_max -= 1
    return img[y_min:y_max,x_min:x_max]

def save_to_video(output_path, output_video_file, frame_rate):
    list_files = os.listdir(output_path)
    list_files.sort()
    # 拿一张图片确认宽高
    img0 = cv2.imread(os.path.join(output_path, list_files[0]))
    # print(img0)
    height, width, layers = img0.shape
    # 视频保存初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
    # 核心，保存的东西
    for f in list_files:
        # print("saving..." + f)
        img = cv2.imread(os.path.join(output_path, f))
        videowriter.write(img)
    videowriter.release()
    cv2.destroyAllWindows()
    print('Success save %s!' % output_video_file)
    pass
# 图片变视频

light_path='/media/jcn/新加卷/JCN/Render/light/train_mid.npy'
transport=np.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_4/val/transport_0.npy')
albedo=np.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_4/val/albedo_0.npy')
out_dir='/home/jcn/桌面/Nerf_TEST/nerf_my/checkpoints_nert_4/Single_view'

light_all = np.load(light_path)#light shape=[40,9,3]
transport=np.reshape(transport,(800,800,9))
albedo=np.reshape(albedo,(800,800,3))

albedo = torch.from_numpy(albedo.astype(np.float32)).clone()
transport = torch.from_numpy(transport.astype(np.float32)).clone()

albedo = albedo.to("cuda")
transport = transport.to("cuda")

tmp_renderings = []
max_val = 0.

n_rotation_div = 72
idx=random.randint(0,100)
for i in range(10):
    light=light_all[idx]
    for j in range(n_rotation_div):
        deg = (360. / n_rotation_div) * j
        R = sh_rot.calc_y_rot(deg / 180. * np.pi)
        coeffs = np.empty_like(light)
        coeffs[:,0] = sh_rot.sh_rot(R, light[:,0])
        coeffs[:,1] = sh_rot.sh_rot(R, light[:,1])
        coeffs[:,2] = sh_rot.sh_rot(R, light[:,2])

        coeffs = torch.from_numpy(coeffs.astype(np.float32)).clone()
        coeffs = coeffs.to("cuda")

        shading = torch.matmul(transport, coeffs)

        # rendering=shading
        # rendering=albedo
        rendering = shading

        tmp_renderings.append(rendering)
        max_val = max((max_val, torch.max(rendering)))
    save_basepath = os.path.join(out_dir,'shading')
    if not os.path.exists(save_basepath):
        os.makedirs(save_basepath)

    for j in range(n_rotation_div):
        rendering = 255 * tmp_renderings[j] / 1
        rendering = rendering.to("cpu")
        rendering = rendering.to('cpu').detach().numpy().copy()
        cv2.imwrite(save_basepath + ('/frame%03d.jpg' % j), rendering[:,:,::-1])

    video_path = save_basepath + '.mp4'
    files_path = save_basepath + '/frame%03d.jpg'
    os.system('ffmpeg -y -r 30 -i ' + files_path + ' -vcodec libx264 -pix_fmt yuv420p -r 60 ' + video_path)

    save_video_path=os.path.join(save_basepath,'shading.mp4')
    save_to_video(output_path=save_basepath,output_video_file=save_video_path, frame_rate=20)

