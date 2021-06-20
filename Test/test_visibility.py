import os, sys
from opt import get_opts
from tqdm import tqdm
import time

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility
from models.rendering_visibility import render_rays, render_rays_visibility_test
from models.sh_util_gpu_n import PRT

# metrics
from metrics import *
from utils import visualize_depth
import numpy as np
import cv2

from termcolor import colored
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

hparams = get_opts()

#Loading Dataset
print(colored('Loading Dataset','red'))
dataset = dataset_dict[hparams.dataset_name]
kwargs = {'root_dir': hparams.root_dir,
          'img_wh': tuple(hparams.img_wh)}
train_dataset = dataset(split='train', **kwargs)
val_dataset = dataset(split='val', **kwargs)

train_data_loader=DataLoader(train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=hparams.batch_size,
                          pin_memory=True)
# Loading Loss
mse_loss=torch.nn.MSELoss(reduction='mean').cuda()

calcu_PRT=PRT().cuda()
calcu_PRT=torch.nn.DataParallel(calcu_PRT)

#Loading Network
print(colored('Loading Network','red'))
corase_Nerf_sigma=NeRF_sigma().cuda()
corase_Nerf_sigma=torch.nn.DataParallel(corase_Nerf_sigma)
fine_Nerf_sigma=NeRF_sigma().cuda()
fine_Nerf_sigma=torch.nn.DataParallel(fine_Nerf_sigma)

vis_Nerf=Visibility().cuda()
vis_Nerf=torch.nn.DataParallel(vis_Nerf)


try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma_23.pt'))
    vis_Nerf.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_visibility/vis_0.pt'))
    print('Loading Pretrained Models ...')
except:
    print('Loading error ! ..')
    sys.exit(1)

embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

#Start Training
print(colored('Start Testing','blue'))
print('total data=',len(train_dataset))
iter_number = int(len(train_dataset) / hparams.batch_size)
print('val data per ', iter_number)

with torch.no_grad():
    for batch in train_data_loader:
        corase_Nerf_sigma.eval()
        fine_Nerf_sigma.eval()
        vis_Nerf.eval()

        rays=batch['rays'].cuda()
        rgbs=batch['rgbs'].cuda()
        masks=batch['masks'].cuda()

        embeddings = [embedding_xyz, embedding_dir]

        # sigma , albedo, surface_points
        results=render_rays(corase_Nerf_sigma,fine_Nerf_sigma,
                            embeddings,rays,
                            hparams.N_samples,
                            hparams.use_disp,
                            hparams.perturb,
                            hparams.noise_std,
                            hparams.N_importance,
                            hparams.chunk_large,  # chunk size is effective in val mode
                            train_dataset.white_back
                            )
        t2=time.time()
        # visibility
        results_vis=render_rays_visibility_test( vis_Nerf,
                                            fine_Nerf_sigma,
                                            embeddings,
                                            calcu_PRT,
                                            surface_points=results['surface_points'],#[N,3]
                                            masks=masks,
                                            directions_num=20,
                                            noise_std=1,
                                            chunk=hparams.chunk,
                                            chunk_large=hparams.chunk_large,
                                            sample_distance=4
                                            )

        sample_vis=results_vis['sample_vis'].detach().cpu().numpy() #[N_rays_choosed,N_directions,1]
        p_vis=results_vis['p_vis'].detach().cpu().numpy()

        sample_vis=np.reshape(sample_vis,(sample_vis.shape[0],20,20,1))
        p_vis=np.reshape(p_vis,(p_vis.shape[0],20,20,1))

        sample_vis=np.repeat(np.asarray(sample_vis*255.0,dtype=np.uint8),3,axis=3)
        p_vis=np.repeat(np.asarray(p_vis*255.0,dtype=np.uint8),3,axis=3)

        for i in range(p_vis.shape[0]):
            temp=np.concatenate([sample_vis,p_vis],axis=1)
            cv2.imwrite(f'/home/jcn/桌面/Nerf/nerf_my/images/{i}.png',temp)








