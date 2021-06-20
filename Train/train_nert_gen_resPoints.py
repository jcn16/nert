import os, sys
from opt import get_opts
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import blender_res
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility,Surface_res
from models.rendering_nert_res import render_rays, render_rays_visibility
from models.sh_util_gpu_nert import PRT
from collections import defaultdict

# metrics
from metrics import *
import numpy as np
import cv2

from termcolor import colored
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
os.environ['CUDA_VISIBLE_DEVICES']='1'
torch.set_printoptions(profile="full")

'''
Using pretained network contained in checkpoints_nert_1

This is to train Surface_res network
'''

hparams = get_opts()

#Loading Dataset
print(colored('Loading Dataset','red'))
kwargs = {'root_dir': hparams.root_dir,
          'img_wh': tuple(hparams.img_wh)}
train_dataset = blender_res.BlenderDataset(split='val', **kwargs)
val_dataset = blender_res.BlenderDataset(split='val', **kwargs)

train_data_loader=DataLoader(train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)
val_data_loader=DataLoader(val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

val_iter=iter(val_data_loader)
# Loading Loss
mse_loss=torch.nn.MSELoss(reduction='mean').cuda()
calcu_PRT=PRT().cuda()
calcu_PRT=torch.nn.DataParallel(calcu_PRT)

#Loading Network
print(colored('Loading Network','red'))
corase_Nerf_sigma=NeRF_sigma().cuda()
corase_Nerf_sigma=torch.nn.DataParallel(corase_Nerf_sigma)
corase_Nerf_albedo=NeRF_albedo_light().cuda()
corase_Nerf_albedo=torch.nn.DataParallel(corase_Nerf_albedo)
fine_Nerf_sigma=NeRF_sigma().cuda()
fine_Nerf_sigma=torch.nn.DataParallel(fine_Nerf_sigma)
fine_Nerf_albedo=NeRF_albedo_light().cuda()
fine_Nerf_albedo=torch.nn.DataParallel(fine_Nerf_albedo)

vis_Nerf=Visibility().cuda()
vis_Nerf=torch.nn.DataParallel(vis_Nerf)

surface_res=Surface_res().cuda()
surface_res=torch.nn.DataParallel(surface_res)

try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_1/corase_sigma_0.pt'))
    corase_Nerf_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_1/corase_albedo_0.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_1/fine_sigma_0.pt'))
    fine_Nerf_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_1/fine_albedo_0.pt'))
    vis_Nerf.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_1/vis_nerf_0.pt'))
    print('Loading sigma ...')
except:
    print('Loading error ..')
    sys.exit(1)


embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

o_shared=torch.optim.Adam([
            {
                "params": surface_res.parameters(),
                "lr": 1e-4,
            },
            {
                "params": corase_Nerf_albedo.parameters(),
                "lr": 0,
            },
            {
                "params": corase_Nerf_sigma.parameters(),
                "lr": 0,
            },
            {
                "params": fine_Nerf_albedo.parameters(),
                "lr": 0,
            },
            {
                "params": fine_Nerf_sigma.parameters(),
                "lr": 0,
            },
            {
                "params": vis_Nerf.parameters(),
                "lr": 0,
            }
        ])

#Loading Checkpoints
if not os.path.exists(hparams.check_dir):
    os.makedirs(hparams.check_dir)
logger=SummaryWriter(os.path.join(hparams.check_dir, 'logs'))
checkpoints_io=CheckpointIO(hparams.check_dir,
                            Surface_res=surface_res,
                            optimizer=o_shared)
try:
    load_dict=checkpoints_io.load('model_latest.pt')
except FileExistsError:
    load_dict=dict()
start_epoch=load_dict.get('epoch_it',0)
it=load_dict.get('it',0)

#Start Training
print(colored('Start Training','blue'))
print('total data=',len(train_dataset))
iter_number = int(len(train_dataset) / hparams.batch_size)
print('val data per ', iter_number)

def choose_points(surface_points, surface_PRT, surface_depth, threshold):
    surface_points = surface_points.view(800, 800, 3).numpy()
    surface_PRT = surface_PRT.view(800, 800, 9).numpy()
    surface_depth = surface_depth.view(800, 800, 1).numpy()

    prt = np.clip(surface_PRT[:, :, 0], 0, 1)
    kernal = np.ones(shape=(3, 3)) / 8
    kernal[1, 1] = 0
    prt_filterd = cv2.filter2D(prt, -1, kernal)
    prt_filterd = np.clip(prt_filterd, 0, 1)

    # compute res
    res = prt_filterd - prt
    res_mask = (res > threshold)

    # saving
    np.save(path + '_points.npy', surface_points)
    np.save(path + '_PRT.npy', surface_PRT)
    np.save(path + '_depth.npy', surface_depth)
    np.save(path + '_choosed.npy', res_mask)

pbar = tqdm(total=len(val_dataset))
#all losse
print('New Epoch!')
for batch in train_data_loader:
    pbar.update(1)
    it += 1

    vis_Nerf.train()
    surface_res.train()

    rays=batch['rays'][0].cuda()
    rgbs=batch['rgbs'][0].cuda()
    masks=batch['masks'][0].cuda()
    path=batch['path'][0]

    embeddings = [embedding_xyz, embedding_dir]

    # calculate surface points, PRT, surface points
    with torch.no_grad():
        corase_Nerf_sigma.eval()
        corase_Nerf_albedo.eval()
        fine_Nerf_sigma.eval()
        fine_Nerf_albedo.eval()
        vis_Nerf.eval()
        surface_PRT=[]
        surface_points=[]
        surface_depth=[]
        for i in range(0,rays.shape[0],hparams.chunk):
            results=render_rays(corase_Nerf_sigma,corase_Nerf_albedo,fine_Nerf_sigma,fine_Nerf_albedo,
                                embeddings,rays[i:i+hparams.chunk],
                                hparams.N_samples,
                                hparams.use_disp,
                                hparams.perturb,
                                hparams.noise_std,
                                hparams.N_importance,
                                hparams.chunk_large,  # chunk size is effective in val mode
                                train_dataset.white_back
                                )
            # visibility
            results_vis=render_rays_visibility( vis_nerf=vis_Nerf,
                                                embeddings=embeddings,
                                                calcu_PRT=calcu_PRT,
                                                surface_points=results['surface_points'],#[N,3]
                                                masks=masks[i:i+hparams.chunk],
                                                directions_num=20,
                                                noise_std=1,
                                                sample_points=hparams.direction_samples,
                                                chunk=hparams.chunk_large,
                                                chunk_large=1024*32,
                                                sample_distance=4,
                                                near=hparams.near,
                                                threshold=hparams.threshold)
            surface_points.append(results['surface_points'].detach().cpu())
            surface_depth.append(results['surface_depth'].detach().cpu())
            surface_PRT.append(results_vis['transport'].detach().cpu())
        surface_points=torch.cat(surface_points,dim=0)
        surface_depth=torch.cat(surface_depth,dim=0)
        surface_PRT=torch.cat(surface_PRT,dim=0)

        choose_points(surface_points,surface_PRT,surface_depth,0.2)










