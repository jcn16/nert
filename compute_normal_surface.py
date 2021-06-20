import os, sys
from opt import get_opts
from tqdm import tqdm
import time

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility
from models.sh_util_gpu_nert import PRT
import numpy as np
import cv2

# metrics
from metrics import *

from termcolor import colored
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES']='1'

hparams = get_opts()

#Loading Dataset
print(colored('Loading Dataset','red'))
dataset = dataset_dict[hparams.dataset_name]
kwargs = {'root_dir': hparams.root_dir,
          'img_wh': tuple(hparams.img_wh)}
train_dataset = dataset(split='train', **kwargs)
train_data_loader=DataLoader(train_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
val_dataset = dataset(split='val', **kwargs)
val_data_loader=DataLoader(val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
test_dataset = dataset(split='test', **kwargs)
test_data_loader=DataLoader(test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

# val_iter=iter(val_data_loader)
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


try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma_23.pt'))
    print('Loading sigma ...')
except:
    print('Loading error ..')
    sys.exit(1)

o_shared=torch.optim.Adam([
            {
                "params": fine_Nerf_sigma.parameters(),
                "lr": 1e-4,
            }
        ])


embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

#Start Training
print(colored('Start Training','blue'))
print('total data=',len(train_dataset))
iter_number = int(len(train_dataset) / hparams.batch_size)
print('val data per ', iter_number)

# To compute gradient
corase_Nerf_sigma.eval()
fine_Nerf_sigma.train()

it=0

dataloaders=[test_data_loader]

for dataloader in dataloaders:
    for val_batch in dataloader:
        it+=1
        rays_all = val_batch['rays'].cuda()
        rays_all =rays_all.squeeze()
        masks_all = val_batch['masks'].cuda()
        masks_all=masks_all.squeeze()
        path=val_batch['path'][0]
        B,_=rays_all.shape

        embeddings = [embedding_xyz, embedding_dir]

        normals = []
        pbar=tqdm(total=int(B/hparams.chunk))
        for i in range(0, B, hparams.chunk):
            pbar.update(1)

            def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
                """
                Sample @N_importance samples from @bins with distribution defined by @weights.

                Inputs:
                    bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
                    weights: (N_rays, N_samples_)
                    N_importance: the number of samples to draw from the distribution
                    det: deterministic or not
                    eps: a small number to prevent division by zero

                Outputs:
                    samples: the sampled samples
                """
                N_rays, N_samples_ = weights.shape
                weights = weights + eps  # prevent division by zero (don't do inplace op!)
                pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
                cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
                cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
                # padded to 0~1 inclusive

                if det:
                    u = torch.linspace(0, 1, N_importance, device=bins.device)
                    u = u.expand(N_rays, N_importance)
                else:
                    u = torch.rand(N_rays, N_importance, device=bins.device)
                u = u.contiguous()

                inds = torch.searchsorted(cdf, u, right=True)
                below = torch.clamp_min(inds - 1, 0)
                above = torch.clamp_max(inds, N_samples_)

                inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
                cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
                bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

                denom = cdf_g[..., 1] - cdf_g[..., 0]
                denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                # anyway, therefore any value for it is fine (set to 1 here)

                samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
                return samples

            def get_surface_points(sigma, points):
                sigma_mask = torch.tensor(sigma > 20, dtype=float)
                sigma_idx = torch.argmax(sigma_mask, axis=1)  # [N_rays,]
                all = []
                for i in range(points.shape[0]):
                    all.append(points[i:i + 1, sigma_idx[i], :])
                all = torch.cat(all, dim=0)
                # all.retain_grad()
                return all

            def inference(nerf_sigma,embedding_xyz, xyz_, dir_, z_vals, weights_only=False):

                N_samples_ = xyz_.shape[1]
                # Embed directions
                xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
                # Perform model inference to get rgb and raw sigma
                B = xyz_.shape[0]
                out_chunks = []

                for i in range(0, B, hparams.chunk):
                    # Embed positions by chunk
                    xyz_embedded = embedding_xyz(xyz_[i:i + hparams.chunk])
                    p_sigma, _ = nerf_sigma(xyz_embedded)
                    out_chunks += [p_sigma]

                out = torch.cat(out_chunks, 0)
                sigmas = out.view(N_rays, N_samples_)

                # Convert these values using volume rendering (Section 4)
                deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
                delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
                deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

                # Multiply each distance by the norm of its corresponding direction ray
                # to convert to real world distance (accounts for non-unit directions).
                deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

                noise = torch.randn(sigmas.shape, device=sigmas.device) * hparams.noise_std

                # compute alpha by the formula (3)
                alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
                alphas_shifted = \
                    torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
                temp = torch.cumprod(alphas_shifted, -1)
                weights = \
                    alphas * temp[:, :-1]  # (N_rays, N_samples_)
                # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
                visibility = temp[:, -1:]
                # from these fomulation, The Weights are really matters, it records all information, with this weights, we can easily get rgbs, z from it.
                depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

                # print('sigam',xyz_.shape,'time=',t1-t0)

                return depth_final, weights, sigmas, visibility

            def inference_surface_points(nerf_sigma,embedding_xyz, xyz_):

                # Embed directions
                xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
                # Perform model inference to get rgb and raw sigma
                B = xyz_.shape[0]
                out_chunks = []

                for i in range(0, B, hparams.chunk):
                    # Embed positions by chunk
                    xyz_embedded = embedding_xyz(xyz_[i:i + hparams.chunk])
                    p_sigma, _ = nerf_sigma(xyz_embedded)
                    out_chunks += [p_sigma]

                out = torch.cat(out_chunks, 0)
                sigmas = out.view(N_rays)

                return  sigmas
            # Extract models from lists
            embedding_xyz = embeddings[0]

            rays=rays_all[i:i+hparams.chunk]

            # Decompose the inputs
            N_rays = rays.shape[0]
            rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
            near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

            # Sample depth points
            z_steps = torch.linspace(0, 1, hparams.N_samples, device=rays.device)  # (N_samples)
            if not hparams.use_disp:  # use linear sampling in depth space
                z_vals = near * (1 - z_steps) + far * z_steps
            else:  # use linear sampling in disparity space
                z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

            z_vals = z_vals.expand(N_rays, hparams.N_samples)

            if hparams.perturb > 0:  # perturb sampling depths (z_vals)
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
                # get intervals between samples
                upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
                lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

                perturb_rand = hparams.perturb * torch.rand(z_vals.shape, device=rays.device)
                z_vals = lower + (upper - lower) * perturb_rand

            xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                                 rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

            # for test

            depth_corase, weights_corase, _, visibility_corase = \
                inference(corase_Nerf_sigma, embedding_xyz, xyz_coarse_sampled, rays_d,
                          z_vals)
            result = {
                      'depth_coarse': depth_corase,
                      'opacity_coarse': weights_corase.sum(1)
                      }

            if hparams.N_importance > 0:  # sample points for fine model
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
                z_vals_ = sample_pdf(z_vals_mid, weights_corase[:, 1:-1],
                                     hparams.N_importance, det=(hparams.perturb == 0)).detach()
                # detach so that grad doesn't propogate to weights_coarse from here

                z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

                xyz_fine_sampled = rays_o.unsqueeze(1) + \
                                   rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                # (N_rays, N_samples+N_importance, 3)
                _, _, sigmas, _ = \
                    inference(fine_Nerf_sigma, embedding_xyz, xyz_fine_sampled, rays_d,
                              z_vals, weights_only=False)

                surface_points_threshold = get_surface_points(sigmas, xyz_fine_sampled)
                surface_points_threshold.requires_grad = True
                surface_points_threshold.retain_grad()

                sigmas_ = inference_surface_points(fine_Nerf_sigma, embedding_xyz, surface_points_threshold)
                o_shared.zero_grad()
                sigmas_.backward(torch.ones_like(sigmas_))

                normal=F.normalize(surface_points_threshold.grad,p=2,dim=1)
                normals.append(normal.detach().cpu().numpy())

        image_normal=np.concatenate(normals,axis=0)
        image_normal=np.reshape(-image_normal,newshape=(800,800,3))
        mask = masks_all.detach().cpu().numpy()
        mask = np.reshape(mask, newshape=(800, 800, 1))
        image_normal=image_normal*mask
        # np.save(path+'_normal_surface.npy',image_normal)

        image_normal=np.asarray((0.5*image_normal+0.5)*255.0*mask,dtype=np.uint8)
        print(path+'_normal_surface.png')
        cv2.imwrite(path+'_normal_surface.png',image_normal[:,:,::-1])


