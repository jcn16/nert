import torch
#from torchsearchsorted import searchsorted
from .prt_util import *
import random
import time

__all__ = ['render_rays']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""

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
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

def get_surface_points(sigma,points):
    sigma_mask=torch.tensor(sigma>20,dtype=float)
    sigma_idx=torch.argmax(sigma_mask,axis=1) #[N_rays,]
    all=torch.zeros(size=(points.shape[0],3),device=sigma.device)
    for i in range(points.shape[0]):
        all[i]=points[i,sigma_idx[i],:]
    return all

def get_surface_depth(suafce_points,cam_loc,directions):
    z_val=(suafce_points-cam_loc)/directions
    z_val=torch.sum(z_val,dim=1,keepdim=True)/3.0 #[N_rays,1]
    return z_val

def render_rays(corase_sigma,
                corase_albedo,
                fine_sigma,
                fine_albedo,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(nerf_sigma, nerf_albedo, embedding_xyz, xyz_, dir_, z_vals, weights_only=False):
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i + chunk])
            p_sigma, features = nerf_sigma(xyz_embedded)
            if not weights_only:
                p_rgbs = nerf_albedo(features)
                out_chunks += [torch.cat([p_rgbs, p_sigma], dim=1)]
            else:
                out_chunks += [p_sigma]

        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        temp = torch.cumprod(alphas_shifted, -1)
        weights = \
            alphas * temp[:, :-1]  # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        visibility = temp[:, -1:]
        if weights_only:
            return weights

        # compute final weighted outputs
        # from these fomulation, The Weights are really matters, it records all information, with this weights, we can easily get rgbs, z from it.
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights, sigmas, visibility

    # Extract models from lists
    embedding_xyz = embeddings[0]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    # for test
    weights_corase= \
        inference(corase_sigma, corase_albedo, embedding_xyz, xyz_coarse_sampled, rays_d,
                  z_vals, weights_only=True)
    result = {
              'opacity_coarse': weights_corase.sum(1)
              }

    if N_importance > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_corase[:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
        # (N_rays, N_samples+N_importance, 3)

        rgb_fine, depth_fine, weights_fine, sigmas, visibility_fine = \
            inference(fine_sigma, fine_albedo, embedding_xyz, xyz_fine_sampled, rays_d,
                      z_vals, weights_only=False)

        surface_points = get_surface_points(sigmas,xyz_fine_sampled)  # [N_rays,3]
        surface_depth=get_surface_depth(surface_points,rays_o,rays_d)

        result['rgb_fine'] = rgb_fine
        # result['depth_fine'] = depth_fine
        result['surface_points'] = surface_points
        result['surface_depth'] = surface_depth

    return result


def render_rays_visibility(
                vis_nerf,
                embeddings,
                calcu_PRT,
                surface_points=None,#[N,3]
                masks=None,
                directions_num=20,
                noise_std=1,
                chunk=1024 * 32,
                chunk_large=1024*32,
                sample_distance=4,
                sample_points=128,
                near=0.01,
                threshold=0.8
                ):

    # Extract models from lists
    # surface_points=surface_points.detach()
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # get directions and embeddings
    directions_, phi, theta = calcu_PRT.module.sampleSphericalDirections_N_rays(surface_points.shape[0],
                                                                                directions_num)  # [N_rays,N_directions,3]

    N_rays, _ = surface_points.shape
    _, N_directions, _ = directions_.shape

    directions_=directions_.view(N_rays*N_directions,3)
    # embedings
    points_embedded=embedding_xyz(surface_points) #[N_rays,points_embeddings]
    direction_embedded=embedding_dir(directions_) #[N_directions,direction_embeddings]
    directions_=directions_.view(N_rays,N_directions,3)

    _,points_dim=points_embedded.shape
    _,direction_dim=direction_embedded.shape

    points_embedded=points_embedded[:,None,:].expand((N_rays,N_directions,points_dim)) #[N_rays,N_directions,_]
    direction_embedded=direction_embedded.view(N_rays,N_directions,direction_dim)

    points_embedded=points_embedded.reshape(-1,points_dim)
    direction_embedded=direction_embedded.reshape(-1,direction_dim)

    # predict visibility from point x , direction w
    out_chunks = []
    for i in range(0, points_embedded.shape[0], chunk_large):
        # Embed positions by chunk
        out_chunks += [vis_nerf(points_embedded[i:i+chunk_large],direction_embedded[i:i+chunk_large])]
    p_visibility=torch.cat(out_chunks,0)
    p_visibility=p_visibility.view(N_rays,N_directions,1)

    # compute PRT
    p_PRT=calcu_PRT.module.computePRT_vis(n=directions_num, order=2, p_vis=p_visibility, phi=phi, theta=theta)  # [N_rays,9]

    result={
        'p_vis':p_visibility,
        'transport':p_PRT
    }

    return result

def render_rays_surface_res(
                embeddings,
                resnet,
                all_rays,
                all_depth,
                masks,
                choosed_points,
                choosed_depth,
                choosed_masks,
                chunk=1024*32
                ):

    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # compute choosed depth
    rays_o=choosed_points[:,0:3]
    rays_d=choosed_points[:,3:6]

    all_res=[]
    for i in range(0,choosed_points.shape[0],chunk):
        p_res=resnet(embedding_xyz(rays_o[i:i+chunk]),embedding_dir(rays_d[i:i+chunk]))
        all_res.append(p_res)

    all_res=torch.cat(all_res,dim=0)
    p_depth=all_depth[choosed_masks]+all_res.squeeze()
    p_depth.retain_grad()

    results={
        'p_depth_inside':p_depth,
        'gt_depth_inside':choosed_depth
    }

    # sample some points out of mask
    rays_idx_all = random.sample(range(all_rays.shape[0]), int(0.2 * all_rays.shape[0]))
    choosed = torch.zeros(all_rays.shape[0], device=all_rays.device)
    choosed[rays_idx_all] = 1
    rays_idx = ((masks.squeeze()<0.5) * choosed > 0)

    choosed_outside=all_rays[rays_idx]
    rays_o = choosed_outside[:, 0:3]
    rays_d = choosed_outside[:, 3:6]

    all_res = []
    for i in range(0, choosed_outside.shape[0], chunk):
        p_res = resnet(embedding_xyz(rays_o[i:i + chunk]), embedding_dir(rays_d[i:i + chunk]))
        all_res.append(p_res)

    all_res = torch.cat(all_res, dim=0)
    results['p_outside_res']=all_res
    results['gt_outside_res']=torch.zeros_like(all_res,device=all_res.device)

    return results


