import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from opt import get_opts
from tqdm import tqdm
import time
import numpy as np

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility
from models.rendering_visibility import render_rays, render_rays_visibility_test
from models.sh_util_gpu_nert import PRT
from collections import defaultdict

# metrics
from metrics import *
import cv2

from termcolor import colored
os.environ['CUDA_VISIBLE_DEVICES']='0'

hparams = get_opts()

#Loading Dataset
print(colored('Loading Dataset','red'))
dataset = dataset_dict[hparams.dataset_name]
kwargs = {'root_dir': hparams.root_dir,
          'img_wh': tuple(hparams.img_wh)}
test_dataset = dataset(split='test', **kwargs)
test_data_loader=DataLoader(test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
train_dataset = dataset(split='train', **kwargs)
train_data_loader=DataLoader(train_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

# Loading Loss
calcu_PRT=PRT().cuda()
calcu_PRT=torch.nn.DataParallel(calcu_PRT)

#Loading Network
print(colored('Loading Network','red'))
corase_Nerf_sigma=NeRF_sigma().cuda()
corase_Nerf_sigma=torch.nn.DataParallel(corase_Nerf_sigma)
fine_Nerf_sigma=NeRF_sigma().cuda()
fine_Nerf_sigma=torch.nn.DataParallel(fine_Nerf_sigma)

try:
    corase_Nerf_sigma.load_state_dict(torch.load('./checkpoints/checkpoints_sigma/corase_sigma_23.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('./checkpoints/checkpoints_sigma/fine_sigma_23.pt'))
    print('Loading sigma ...')
except:
    print('Loading error ..')
    # sys.exit(1)


embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)


def save_obj_mesh(mesh_path, verts):
    file = open(mesh_path, 'w')
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()

#Start Training
print(colored('Start Test','blue'))

with torch.no_grad():
    corase_Nerf_sigma.eval()
    fine_Nerf_sigma.eval()

    dataloaders=[train_data_loader,test_data_loader]
    for dataloader in dataloaders:
        for val_batch in dataloader:

            path=val_batch['path'][0]
            rays = val_batch['rays'].cuda()
            rays =rays.squeeze()
            rgbs = val_batch['rgbs'].cuda()
            rgbs=rgbs.squeeze()
            masks = val_batch['masks'].cuda()
            masks=masks.squeeze(0)
            surface_points_all = val_batch['surfaces'][0].cuda()

            B,_=rays.shape

            results = defaultdict(list)
            pbar=tqdm(total=(B//hparams.chunk))

            '''
            chunk=1024,near, threshold
            '''
            for i in range(0, B, hparams.chunk):
                pbar.update(1)
                surface_points=surface_points_all[i:i+hparams.chunk]

                def inference(fine_sigma, embedding_xyz, xyz_, z_vals):

                    N_samples_ = xyz_.shape[1]
                    # Embed directions
                    xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)

                    # Perform model inference to get rgb and raw sigma
                    B = xyz_.shape[0]
                    out_chunks = []
                    for i in range(0, B, hparams.chunk_large):
                        # Embed positions by chunk
                        xyzdir_embedded = embedding_xyz(xyz_[i:i + hparams.chunk_large])
                        out_chunks += [fine_sigma(xyzdir_embedded, vis_predict=True)]

                    out = torch.cat(out_chunks, 0)
                    sigmas = out.view(N_rays_choosed * N_directions, N_samples_)

                    # Convert these values using volume rendering (Section 4)
                    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
                    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
                    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

                    # Multiply each distance by the norm of its corresponding direction ray
                    # to convert to real world distance (accounts for non-unit directions).

                    noise = torch.randn(sigmas.shape, device=sigmas.device) * hparams.noise_std

                    # compute alpha by the formula (3)
                    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
                    alphas_shifted = \
                        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
                    temp = torch.cumprod(alphas_shifted, -1)
                    weights = \
                        alphas * temp[:, :-1]  # (N_rays, N_samples_)
                    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
                    return weights, temp[:, -1:]

                # get directions and embeddings
                directions_, phi, theta = calcu_PRT.module.sampleSphericalDirections_N_rays(surface_points.shape[0],20)  # [N_rays,N_directions,3]

                N_rays, _ = surface_points.shape
                _, N_directions, _ = directions_.shape

                directions_ = directions_.view(N_rays * N_directions, 3)

                # xyz and directions embedings without disturb
                points_embedded = embedding_xyz(surface_points)  # [N_rays,points_embeddings]
                direction_embedded = embedding_dir(directions_)  # [N_directions,direction_embeddings]
                directions_ = directions_.view(N_rays, N_directions, 3)

                _, points_dim = points_embedded.shape
                _, direction_dim = direction_embedded.shape

                points_embedded = points_embedded[:, None, :].expand(
                    (N_rays, N_directions, points_dim))  # [N_rays,N_directions,_]
                direction_embedded = direction_embedded.view(N_rays, N_directions, direction_dim)

                points_embedded = points_embedded.reshape(-1, points_dim)
                direction_embedded = direction_embedded.reshape(-1, direction_dim)

                # random choose some rays to calculate GT
                corase_sample = hparams.direction_samples
                points_s = surface_points[:, None, None, :].expand(N_rays, N_directions, corase_sample, 3)
                directions_s = directions_[:, :, None, :].expand(N_rays, N_directions, corase_sample, 3)
                z_steps = torch.linspace(0, 1, corase_sample, device=surface_points.device)  # (N_samples)
                z_vals = hparams.near * (1 - z_steps) + 4 * z_steps
                z_vals = z_vals[None, None, :, None].expand(N_rays, N_directions, corase_sample, 1)

                corase_points = points_s + directions_s * z_vals  # [N_rays,N_directions,corase_sample,3]

                choose_idx = (masks[i:i+hparams.chunk].squeeze() > 0)  # [N_rays_choosed]
                vis = torch.zeros(size=(N_rays, N_directions, 1), device=corase_points.device)

                corase_points_choosed = corase_points[choose_idx]
                z_vals_choosed = z_vals[choose_idx]

                N_rays_choosed = z_vals_choosed.shape[0]

                corase_points_choosed = corase_points_choosed.reshape(N_rays_choosed * N_directions, corase_sample, 3)
                z_vals_choosed = z_vals_choosed.reshape((N_rays_choosed * N_directions, corase_sample))

                if corase_points_choosed.shape[0] > 0:
                    _, sample_visibility = inference(fine_Nerf_sigma, embedding_xyz, corase_points_choosed, z_vals_choosed)
                    sample_visibility = sample_visibility.reshape(N_rays_choosed, N_directions, 1)
                    vis[choose_idx] = sample_visibility

                gt_PRT = calcu_PRT.module.computePRT_vis(n=20, order=2, p_vis=vis, phi=phi,
                                                         theta=theta)  # [N_rays,9]

                results['transport']+=[gt_PRT.detach().cpu()]

            for k, v in results.items():
                results[k] = torch.cat(v, 0)

            gt_PRT=results['transport']
            W, H = hparams.img_wh
            gt_transport = gt_PRT.view(H, W, 9)

            gt_transport = gt_transport.numpy()  # [H,W,9]
            np.save(path+'_transport.npy', gt_transport)










