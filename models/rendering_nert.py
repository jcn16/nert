import torch
#from torchsearchsorted import searchsorted
from .prt_util import *
from .sh_util import *
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
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
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

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['surface_points'] = surface_points
        # result['sigmas'] = sigmas

    return result


def render_rays_visibility(
                vis_nerf,
                fine_sigma,
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
    '''
    This is for test_visibility.py
    input: surface points from render_rays
    output: fine_sigma——>sample_vis
            vis_nerf——>p_vis
    The difference is that ALL directions are used
    '''

    def inference(fine_sigma, embedding_xyz, xyz_,z_vals):

        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk_large):
            # Embed positions by chunk
            xyzdir_embedded = embedding_xyz(xyz_[i:i + chunk_large])
            out_chunks += [fine_sigma(xyzdir_embedded, vis_predict=True)]

        out = torch.cat(out_chunks, 0)
        sigmas = out.view(N_rays_choosed*N_directions_choosed, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        temp=torch.cumprod(alphas_shifted, -1)
        weights = \
            alphas * temp[:, :-1]  # (N_rays, N_samples_)
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        return weights,temp[:,-1:]

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

    old_points=points_embedded
    old_points.retain_grad()

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
    p_PRT.retain_grad()

    # calculating GT visibility using Fine NeRF
    # random choose some rays to calculate GT
    corase_sample=sample_points
    points_s=surface_points[:,None,None,:].expand(N_rays,N_directions,corase_sample,3)
    directions_s=directions_[:,:,None,:].expand(N_rays,N_directions,corase_sample,3)
    z_steps = torch.linspace(0, 1, corase_sample, device=surface_points.device)  # (N_samples)
    z_vals = near * (1 - z_steps) + sample_distance * z_steps
    z_vals = z_vals[None,None,:,None].expand(N_rays,N_directions,corase_sample,1)

    corase_points=points_s+directions_s*z_vals #[N_rays,N_directions,corase_sample,3]

    '''
    random choose rays INSIDE and directions to supervise
    '''
    rays_idx_all=random.sample(range(N_rays),int(0.6*N_rays))
    directions_idx=random.sample(range(N_directions),int(0.5*N_directions))

    # choose inside object points
    choosed=torch.zeros(N_rays,device=corase_points.device)
    choosed[rays_idx_all]=1

    rays_idx=((masks.squeeze())*choosed>0)

    # choosed points Through corase network
    choosed_points=corase_points[rays_idx,:,:,:]
    choosed_points=choosed_points[:,directions_idx,:,:]
    N_rays_choosed, N_directions_choosed,_,_=choosed_points.shape
    choosed_points=choosed_points.reshape(N_rays_choosed*N_directions_choosed,corase_sample,3)

    choosed_z_vals=z_vals[rays_idx,:,:,:]
    choosed_z_vals=choosed_z_vals[:,directions_idx,:,:]
    choosed_z_vals=choosed_z_vals.reshape((N_rays_choosed*N_directions_choosed,corase_sample))

    if choosed_points.shape[0] > 0:
        _, sample_visibility_inside = inference(fine_sigma, embedding_xyz, choosed_points, choosed_z_vals)
        sample_visibility_inside = sample_visibility_inside.reshape(N_rays_choosed, N_directions_choosed, 1)
        p_visibility_inside = p_visibility[rays_idx, :, :]
        p_visibility_inside = p_visibility_inside[:, directions_idx, :]
    else:
        # inplace module, just in case all points are outside
        sample_visibility_inside = torch.ones(size=(1, 1, 1),
                                              device=p_visibility.device)
        p_visibility_inside = torch.ones(size=(1, 1, 1),
                                         device=p_visibility.device)

    '''
        random choose rays OUTSIDE and directions to supervise, sample points are less than inside
    '''
    rays_idx_all = random.sample(range(N_rays), int(0.2 * N_rays))
    directions_idx = random.sample(range(N_directions), int(0.2 * N_directions))

    # choose inside object points
    choosed = torch.zeros(N_rays, device=corase_points.device)
    choosed[rays_idx_all] = 1

    rays_idx = ((torch.tensor(masks.squeeze()<0.5,dtype=float)) * choosed > 0)

    temp=p_visibility[rays_idx,:,:]
    temp=temp[:,directions_idx,:]
    if temp.shape[0] > 0:
        sample_visibility_outside = torch.zeros(size=(temp.shape[0], temp.shape[1], 1),
                                                device=p_visibility.device)
        p_visibility_outside = p_visibility[rays_idx, :, :]
        p_visibility_outside = p_visibility_outside[:, directions_idx, :]
    else:
        # inplace module, just in case all points are outside
        sample_visibility_outside = torch.ones(size=(1, 1, 1),
                                               device=p_visibility.device)
        p_visibility_outside = torch.ones(size=(1, 1, 1),
                                          device=p_visibility.device)


    result={
        'sample_outside':sample_visibility_outside,
        'p_outside':p_visibility_outside,
        'sample_inside': sample_visibility_inside,
        'p_inside': p_visibility_inside,
        'p_vis':p_visibility,
        'transport':p_PRT,
        'points_embedded':old_points
    }

    return result
