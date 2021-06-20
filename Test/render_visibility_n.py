import os, sys
from opt import get_opts
from tqdm import tqdm
import time
import numpy as np

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility
from models.rendering_visibility import render_rays, render_sample_visibility_n
from models.sh_util_gpu_n import PRT
from collections import defaultdict

# metrics
from metrics import *
from utils import visualize_depth
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

corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))

try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma_23.pt'))
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

    '''
    chunk=1024,
    '''
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
        rendered_ray_vis = \
            render_sample_visibility_n(
                                    fine_Nerf_sigma,
                                    embeddings,
                                    calcu_PRT,
                                    surface_points=rendered_ray_chunks['surface_points'],#[N,3]
                                    masks=masks[i:i+hparams.chunk],
                                    directions_num=20,
                                    noise_std=1,
                                    chunk=hparams.chunk_inside,
                                    sample_distance=4,
                                    direction_samples=hparams.direction_samples,
                                    near=hparams.near,
                                    threshold=hparams.threshold
                                    )
        results['transport']+=[rendered_ray_vis['transport']]
        results['vis']+=[rendered_ray_vis['vis']]

        # save visibility map per pixel
        # sample_points=rendered_ray_vis['sample_visibility']
        # if rendered_ray_vis['show']:
        # if rendered_ray_vis['show']:
        #     for n in range(sample_points.shape[0]):
        #         temp=torch.reshape(sample_points[n],shape=(20,20,1))
        #         temp=temp.detach().cpu().numpy()
        #         temp=np.repeat(temp,3,axis=2)
        #         temp=np.asarray(temp*255.0,np.uint8)
        #         cv2.imwrite(f'/home/jcn/桌面/Nerf/nerf_my/images/{i}_vis_{n}.png',temp)

    for k, v in results.items():
        results[k] = torch.cat(v, 0)

    p_PRT=results['transport']
    W, H = hparams.img_wh
    p_transport = p_PRT.view(H, W, 9)
    p_transport = p_transport.permute(2, 0, 1)  # (9, H, W)
    all=[]
    for i in range(9):
        all.append((0.5*(torch.repeat_interleave(p_transport[i:i+1,:,:],3,dim=0))+0.5)*255.0)

    show_img = torch.cat(all, 2)
    logger.add_image('val/show_transport', show_img.byte(), 1)

    rgbs=rgbs.view(H,W,3)
    rgbs=rgbs.permute(2,0,1)
    rgbs=rgbs*255.0
    logger.add_image('val/image',rgbs.byte(),1)

    # save visibility, size=[
    vis=results['vis'].squeeze().view(H,W,400).detach().cpu().numpy()
    np.save('/home/jcn/桌面/Nerf/nerf_my/visibility.npy',vis)

    # save images
    rgbs=rgbs.permute(1,2,0).cpu().numpy()
    rgbs=np.asarray(rgbs,dtype=np.uint8)
    cv2.imwrite('/normals/rgbs.png', rgbs)

    p_transport=p_transport.permute(1,2,0).cpu().numpy() #[H,W,9]
    np.save('/home/jcn/桌面/Nerf/nerf_my/transport.npy', p_transport)
    # p_transport=np.asarray(p_transport,dtype=np.uint8)
    for i in range(9):
        temp=np.asarray(np.clip((0.5*(np.repeat(p_transport[:,:,i:i+1],3,axis=2))+0.5)*255.0,0,255),dtype=np.uint8)
        cv2.imwrite(f'/normals/transport_{i}.png', temp)










