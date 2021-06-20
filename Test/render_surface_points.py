import os, sys
from opt import get_opts
from tqdm import tqdm
import time
import numpy as np

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Surface_res
from models.rendering_visibility import render_rays, render_rays_visibility, render_sample_visibility
from models.sh_util_gpu_nert import PRT
from collections import defaultdict

# metrics
from metrics import *
from utils import visualize_depth
import cv2

from termcolor import colored
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
os.environ['CUDA_VISIBLE_DEVICES']='1'

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
fine_Nerf_sigma=NeRF_sigma().cuda()
fine_Nerf_sigma=torch.nn.DataParallel(fine_Nerf_sigma)

surface_res=Surface_res().cuda()
surface_res=torch.nn.DataParallel(surface_res)

try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma_23.pt'))
    surface_res.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_res/surface_res_0.pt'))
    print('Loading sigma ...')
except:
    print('Loading error ..')
    # sys.exit(1)


embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

#Loading Checkpoints
if not os.path.exists(hparams.check_dir):
    os.makedirs(hparams.check_dir)
logger=SummaryWriter(os.path.join(hparams.check_dir, 'logs'))
checkpoints_io=CheckpointIO(hparams.check_dir,
                            corase_Nerf_sigma=corase_Nerf_sigma,
                            fine_Nerf_sigma=fine_Nerf_sigma
                            )
try:
    load_dict=checkpoints_io.load('model_latest.pt')
except FileExistsError:
    load_dict=dict()
start_epoch=load_dict.get('epoch_it',0)
it=load_dict.get('it',0)


def save_obj_mesh(mesh_path, verts):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()

def get_surface_points(sigma,points):
    sigma_mask=torch.tensor(sigma>20,dtype=float)
    sigma_idx=torch.argmax(sigma_mask,axis=1) #[N_rays,]
    all=torch.zeros(size=(points.shape[0],3))
    for i in range(points.shape[0]):
        all[i]=points[i,sigma_idx[i],:]
    return all

#Start Training
print(colored('Start Training','blue'))
print('total data=',len(train_dataset))
iter_number = int(len(train_dataset) / hparams.batch_size)
print('val data per ', iter_number)

try:
    val_batch = val_iter.__next__()
except StopIteration:
    val_iter = iter(val_data_loader)
    val_batch = val_iter.__next__()

using_res = True

with torch.no_grad():
    corase_Nerf_sigma.eval()
    fine_Nerf_sigma.eval()

    rays = val_batch['rays'].cuda()
    rays =rays.squeeze()
    rgbs = val_batch['rgbs'].cuda()
    rgbs=rgbs.squeeze()
    masks = val_batch['masks'].cuda()
    masks=masks.squeeze(0)
    B,_=rays.shape

    embeddings = [embedding_xyz, embedding_dir]

    results = defaultdict(list)
    pbar=tqdm(total=(B//hparams.chunk))

    for i in range(0, B, hparams.chunk):
        pbar.update(1)
        rendered_ray_chunks = \
            render_rays(corase_Nerf_sigma,fine_Nerf_sigma,
                        embeddings,rays[i:i+hparams.chunk],
                        hparams.N_samples,
                        hparams.use_disp,
                        hparams.perturb,
                        hparams.noise_std,
                        hparams.N_importance,
                        hparams.chunk_inside,  # chunk size is effective in val mode
                        train_dataset.white_back
                        )
        if using_res:
            res=surface_res.forward(input_xyz=embedding_xyz(rendered_ray_chunks['surface_points_threshold']),
                                               input_dir=embedding_dir(rays[i:i+hparams.chunk, 3:6]))
            results['surface_points_threshold']+=[(rendered_ray_chunks['surface_points_threshold']+res).detach().cpu()]
        else:
            results['surface_points_threshold']+=[rendered_ray_chunks['surface_points_threshold'].detach().cpu()]
        results['surface_points_depth']+=[rendered_ray_chunks['surface_points_depth'].detach().cpu()]
        results['sigmas']+=[torch.relu(rendered_ray_chunks['sigmas'].detach().cpu())] #[N_rays,N_sample]
        # results['xyz']+=[rendered_ray_chunks['xyz']] #[N_rays,N_sample,3]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    all=[]
    for i in range(results['surface_points_depth'].shape[0]):
        if results['surface_points_depth'][i,2]>results['surface_points_threshold'][i,2]:
            all.append(results['surface_points_threshold'])

    if using_res:
        points = results['surface_points_threshold']
        save_obj_mesh('./meshs/surface_threshold_res.obj', points)
        print('Save Successful')
    else:
        points=results['surface_points_depth']
        save_obj_mesh('./meshs/surface_depth.obj',points)
        print('Save Successful')
        points=results['surface_points_threshold']
        save_obj_mesh('./meshs/surface_threshold.obj',points)
        print('Save Successful')

    # sigmas=results['sigmas'].detach().cpu()
    # xyz=results['xyz'].detach().cpu()
    # all=get_surface_points(sigmas,xyz)
    # print(sigmas.max(),sigmas.min())
    # save_obj_mesh('./meshs/surface_relu.obj', all)
    # print('Save Successful')











