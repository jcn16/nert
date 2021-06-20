import torch
#from torchsearchsorted import searchsorted
from .prt_util import *
from .sh_util_gpu_nert import *
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
    all=[]
    for i in range(points.shape[0]):
        all.append(points[i:i+1,sigma_idx[i],:])
    all=torch.cat(all,dim=0)
    # all.retain_grad()
    return all

def render_rays(corase_sigma,
                fine_sigma,
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
    '''
    input: coarse_sigma, fine_sigma are used to compute volume density
    output: sigma
    '''

    def inference(nerf_sigma,embedding_xyz, xyz_, dir_, z_vals, weights_only=False):

        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []

        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i + chunk])
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

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

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

    depth_corase, weights_corase, _, visibility_corase = \
        inference(corase_sigma, embedding_xyz, xyz_coarse_sampled, rays_d,
                  z_vals)
    result = {
              'depth_coarse': depth_corase,
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

        depth_fine, weights_fine,sigmas, visibility_fine = \
            inference(fine_sigma, embedding_xyz, xyz_fine_sampled, rays_d,
                      z_vals, weights_only=False)

        surface_points_depth = rays_o + rays_d * (depth_fine.unsqueeze(1))  # [N_rays,3]
        surface_points_threshold = get_surface_points(sigmas,xyz_fine_sampled)

        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)
        result['surface_points_depth'] = surface_points_depth
        result['surface_points_threshold'] = surface_points_threshold
        result['visibility'] = visibility_fine  # [N,1]
        result['sigmas']=sigmas

    return result

def render_rays_visibility_test(
                fine_sigma,
                embeddings,
                calcu_PRT,
                surface_points=None,#[N,3]
                masks=None,
                directions_num=20,
                noise_std=1,
                direction_samples=128,
                chunk_large=1024*32,
                sample_distance=4,
                near=0.01,
                threshold=0.5
                ):

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
        sigmas = out.view(N_rays_choosed*N_directions, N_samples_)

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

    # xyz and directions embedings without disturb
    points_embedded=embedding_xyz(surface_points) #[N_rays,points_embeddings]
    direction_embedded=embedding_dir(directions_) #[N_directions,direction_embeddings]
    directions_=directions_.view(N_rays,N_directions,3)

    _,points_dim=points_embedded.shape
    _,direction_dim=direction_embedded.shape

    points_embedded=points_embedded[:,None,:].expand((N_rays,N_directions,points_dim)) #[N_rays,N_directions,_]
    direction_embedded=direction_embedded.view(N_rays,N_directions,direction_dim)

    points_embedded=points_embedded.reshape(-1,points_dim)
    direction_embedded=direction_embedded.reshape(-1,direction_dim)

    # predict visibility from point x , direction w， without disturb
    out_chunks = []
    for i in range(0, points_embedded.shape[0], chunk_large):
        # Embed positions by chunk
        out_chunks += [vis_nerf(points_embedded[i:i+chunk_large],direction_embedded[i:i+chunk_large])]
    p_visibility=torch.cat(out_chunks,0)
    p_visibility=p_visibility.view(N_rays,N_directions,1)
    # calculate transport map

    p_PRT = calcu_PRT.module.computePRT_vis(n=directions_num, order=2, p_vis=p_visibility, phi=phi, theta=theta)  # [N_rays,9]

    # calculating GT visibility using Fine NeRF
    # random choose some rays to calculate GT
    corase_sample=direction_samples
    points_s=surface_points[:,None,None,:].expand(N_rays,N_directions,corase_sample,3)
    directions_s=directions_[:,:,None,:].expand(N_rays,N_directions,corase_sample,3)
    z_steps = torch.linspace(0, 1, corase_sample, device=surface_points.device)  # (N_samples)
    z_vals = near * (1 - z_steps) + sample_distance * z_steps
    z_vals = z_vals[None,None,:,None].expand(N_rays,N_directions,corase_sample,1)

    corase_points=points_s+directions_s*z_vals #[N_rays,N_directions,corase_sample,3]

    choose_idx = (masks.squeeze() > 0)  # [N_rays_choosed]
    vis = torch.zeros(size=(N_rays, N_directions, 1), device=corase_points.device)

    corase_points_choosed = corase_points[choose_idx]
    z_vals_choosed = z_vals[choose_idx]

    N_rays_choosed = z_vals_choosed.shape[0]

    corase_points_choosed = corase_points_choosed.reshape(N_rays_choosed * N_directions, corase_sample, 3)
    z_vals_choosed = z_vals_choosed.reshape((N_rays_choosed * N_directions, corase_sample))

    if corase_points_choosed.shape[0]>0:
        _,sample_visibility=inference(fine_sigma,embedding_xyz,corase_points_choosed,z_vals_choosed)
        sample_visibility=sample_visibility.reshape(N_rays_choosed,N_directions,1)
        vis[choose_idx]=sample_visibility

    gt_PRT = calcu_PRT.module.computePRT_vis(n=directions_num, order=2, p_vis=vis, phi=phi, theta=theta)  # [N_rays,9]

    # print('vis','data=',t1-t0,'vis_pred=',t2-t1,'sigma=',t3-t2)
    result={
        'p_transport': p_PRT,
        'gt_transport': gt_PRT
    }

    return result

def render_rays_visibility(vis_nerf,
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
                direction_samples=128,
                near=0.01
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
    # xyz and directions embedings,without disturb
    points_embedded=embedding_xyz(surface_points) #[N_rays,points_embeddings]
    direction_embedded=embedding_dir(directions_) #[N_directions,direction_embeddings]
    directions_=directions_.view(N_rays,N_directions,3)

    _,points_dim=points_embedded.shape
    _,direction_dim=direction_embedded.shape

    points_embedded=points_embedded[:,None,:].expand((N_rays,N_directions,points_dim)) #[N_rays,N_directions,_]
    direction_embedded=direction_embedded.view(N_rays,N_directions,direction_dim)

    points_embedded=points_embedded.reshape(-1,points_dim)
    direction_embedded=direction_embedded.reshape(-1,direction_dim)

    # predict visibility from point x , direction w, without disturb
    out_chunks = []
    for i in range(0, points_embedded.shape[0], chunk):
        # Embed positions by chunk
        out_chunks += [vis_nerf(points_embedded[i:i+chunk],direction_embedded[i:i+chunk])]
    p_visibility=torch.cat(out_chunks,0)
    p_visibility=p_visibility.view(N_rays,N_directions,1)

    # xyz and directions embedings WITH disturb
    xyz_noise = torch.normal(mean=0, std=0.0001, size=surface_points.shape,device=p_visibility.device)
    assert xyz_noise.shape == surface_points.shape, "Noise generating is wrong!"
    surface_points_disturb = surface_points + xyz_noise
    points_embedded_disturb = embedding_xyz(surface_points_disturb)  # [N_rays,points_embeddings]
    points_embedded_disturb = points_embedded_disturb[:, None, :].expand(
        (N_rays, N_directions, points_dim))  # [N_rays,N_directions,_]
    points_embedded_disturb = points_embedded_disturb.reshape(-1, points_dim)

    # predict visibility from point x , direction w， with disturb
    out_chunks_disturb = []
    for i in range(0, points_embedded_disturb.shape[0], chunk_large):
        # Embed positions by chunk
        out_chunks_disturb += [
            vis_nerf(points_embedded_disturb[i:i + chunk_large], direction_embedded[i:i + chunk_large])]
    p_visibility_disturb = torch.cat(out_chunks_disturb, 0)
    p_visibility_disturb = p_visibility_disturb.view(N_rays, N_directions, 1)

    # calculating GT visibility using Fine NeRF
    # random choose some rays to calculate GT
    corase_sample=direction_samples
    points_s=surface_points[:,None,None,:].expand(N_rays,N_directions,corase_sample,3)
    directions_s=directions_[:,:,None,:].expand(N_rays,N_directions,corase_sample,3)
    z_steps = torch.linspace(0, 1, corase_sample, device=surface_points.device)  # (N_samples)
    z_vals = near * (1 - z_steps) + sample_distance * z_steps
    z_vals = z_vals[None,None,:,None].expand(N_rays,N_directions,corase_sample,1)

    corase_points=points_s+directions_s*z_vals #[N_rays,N_directions,corase_sample,3]

    '''
        random choose rays INSIDE and directions to supervise
        '''
    rays_idx_all = random.sample(range(N_rays), int(0.6 * N_rays))
    directions_idx = random.sample(range(N_directions), int(0.5 * N_directions))

    # choose inside object points
    choosed = torch.zeros(N_rays, device=corase_points.device)
    choosed[rays_idx_all] = 1

    rays_idx = ((masks.squeeze()) * choosed > 0)

    # choosed points Through corase network
    choosed_points = corase_points[rays_idx, :, :, :]
    choosed_points = choosed_points[:, directions_idx, :, :]
    N_rays_choosed, N_directions_choosed, _, _ = choosed_points.shape
    choosed_points = choosed_points.reshape(N_rays_choosed * N_directions_choosed, corase_sample, 3)

    choosed_z_vals = z_vals[rays_idx, :, :, :]
    choosed_z_vals = choosed_z_vals[:, directions_idx, :, :]
    choosed_z_vals = choosed_z_vals.reshape((N_rays_choosed * N_directions_choosed, corase_sample))

    if choosed_points.shape[0] > 0:
        _, sample_visibility_inside = inference(fine_sigma, embedding_xyz, choosed_points, choosed_z_vals)
        sample_visibility_inside = sample_visibility_inside.reshape(N_rays_choosed, N_directions_choosed, 1)
        p_visibility_inside = p_visibility[rays_idx, :, :]
        p_visibility_inside = p_visibility_inside[:, directions_idx, :]
    else:
        # inplace module, just in case all points are outside
        sample_visibility_inside = torch.ones(size=(10, 10, 1),
                                                device=p_visibility.device)
        p_visibility_inside = torch.ones(size=(10, 10, 1),
                                              device=p_visibility.device)

    '''
        random choose rays OUTSIDE and directions to supervise, sample points are less than inside
    '''
    rays_idx_all = random.sample(range(N_rays), int(0.2 * N_rays))
    directions_idx = random.sample(range(N_directions), int(0.2 * N_directions))

    choosed = torch.zeros(N_rays, device=corase_points.device)
    choosed[rays_idx_all] = 1

    rays_idx = ((torch.tensor(masks.squeeze() < 0.5, dtype=float)) * choosed > 0)
    temp = p_visibility[rays_idx, :, :]
    temp = temp[:, directions_idx, :]
    if rays_idx.shape[0]>0:
        sample_visibility_outside = torch.zeros(size=(temp.shape[0], temp.shape[1], 1), device=p_visibility.device)
        p_visibility_outside = p_visibility[rays_idx, :, :]
        p_visibility_outside = p_visibility_outside[:, directions_idx, :]
    else:
        # inplace module, just in case all points are outside
        sample_visibility_outside = torch.ones(size=(10, 10, 1),
                                              device=p_visibility.device)
        p_visibility_outside = torch.ones(size=(10, 10, 1),
                                         device=p_visibility.device)

    result = {
        'sample_outside': sample_visibility_outside,
        'p_outside': p_visibility_outside,
        'sample_inside': sample_visibility_inside,
        'p_inside': p_visibility_inside,
        'p_vis': p_visibility,
        'p_vis_disturb':p_visibility_disturb
    }

    return result

def render_sample_visibility(
                fine_sigma,
                embeddings,
                calcu_PRT,
                surface_points=None,#[N,3]
                masks=None,
                directions_num=20,
                noise_std=1,
                chunk=1024 * 32,
                sample_distance=4,
                direction_samples=128
                ):

    def inference(fine_sigma, embedding_xyz, xyz_,z_vals):
        # print('starting infer')
        t0=time.time()
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        # print(N_rays_choosed,N_directions,xyz_.shape)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyzdir_embedded = embedding_xyz(xyz_[i:i + chunk])

            out_chunks += [fine_sigma(xyzdir_embedded, vis_predict=True)]

        out = torch.cat(out_chunks, 0)
        sigmas = out.view(N_rays_choosed*N_directions, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).

        # compute alpha by the formula (3)
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        temp=torch.cumprod(alphas_shifted, -1)
        weights = \
            alphas * temp[:, :-1]  # (N_rays, N_samples_)
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        t1=time.time()
        # print('ending infer time=',t1-t0)
        return weights,temp[:,-1:]

    def computeNormal(depth,h,w):
        new_depth=depth.view(1,1,h,w)
        padding=torch.nn.ReflectionPad2d(1)
        new_depth=padding(new_depth) #[1,1,h+2,w+2]
        depth_pad=new_depth.squeeze()#[h+2,w+2]

        dzdx = (depth_pad[2:2 + h, :] - depth_pad[0:0 + h, :]) / 2.0  # [h,w]
        dzdy = (depth_pad[:, 2:2 + w] - depth_pad[:, 0:0 + w]) / 2.0  # [h,w]
        z = torch.ones_like(depth)
        z= z.cuda()
        normal = torch.cat([dzdx[:, :, None], dzdy[:, :, None], z[:, :, None]], axis=1)
        normal_sum = torch.sqrt(torch.power(dzdx, 2) + torch.power(dzdy, 2) + torch.power(z, 2))
        normal = normal / normal_sum[:, :, None]
        return normal

    def computeNormal_points(points):
        '''
        depth:[h,w,3]
        output:[h,w,3]
        '''
        h, w, _ = points.shape
        padding = torch.nn.ReflectionPad2d(1)
        new_points=points.permute(2,0,1).unsqueeze(0) #[1,3,h,w]
        depth_pad = padding(new_points)
        depth_pad =depth_pad.squeeze(0).permute(1,2,0)

        left_y = depth_pad[2:2 + h, 1:1 + w] - depth_pad[1:1 + h, 1:1 + w]  # [h,w,3]
        left_x = depth_pad[1:1 + h, 0:0 + w] - depth_pad[1:1 + h, 1:1 + w]  # [h,w,3]
        right_x = depth_pad[1:1 + h, 2:2 + w] - depth_pad[1:1 + h, 1:1 + w]
        right_y = depth_pad[0:0 + h, 1:1 + w] - depth_pad[1:1 + h, 1:1 + w]

        face_1 = (torch.cross(right_x.view(-1, 3), right_y.view(-1, 3))).view(h, w, 3)
        face_1 = face_1 / torch.sqrt(torch.sum(torch.power(face_1, 2), dim=2, keepdims=True))

        face_2 = (torch.cross(right_y.view(-1, 3), left_x.view(-1, 3))).view(h, w, 3)
        face_2 = face_2 / torch.sqrt(torch.sum(torch.power(face_2, 2), dim=2, keepdims=True))

        face_3 = (torch.cross(left_x.view(-1, 3), left_y.view(-1, 3))).view(h, w, 3)
        face_3 = face_3 / torch.sqrt(torch.sum(torch.power(face_3, 2), dim=2, keepdims=True))

        face_4 = (torch.cross(left_y.view(-1, 3), right_x.view(-1, 3))).view(h, w, 3)
        face_4 = face_4 / torch.sqrt(torch.sum(torch.power(face_4, 2), dim=2, keepdims=True))

        normal = (face_1 + face_2 + face_3 + face_4) / 4.0
        normal = normal / torch.sqrt(torch.sum(torch.power(normal, 2), dim=2, keepdims=True))
        return normal

    # Extract models from lists
    surface_points=surface_points.detach()
    embedding_xyz = embeddings[0]

    # get directions and embeddings
    directions_,phi, theta=calcu_PRT.module.sampleSphericalDirections(directions_num) #[N,3]

    N_rays, _ = surface_points.shape
    N_directions, _ = directions_.shape

    # calculating GT visibility using corase NeRF
    # random choose some rays to calculate GT
    corase_sample=direction_samples
    # fine_sample=64
    points_s=surface_points[:,None,None,:].expand(N_rays,N_directions,corase_sample,3)
    directions_s=directions_[None,:,None,:].expand(N_rays,N_directions,corase_sample,3)
    z_steps = torch.linspace(0, 1, corase_sample, device=surface_points.device)  # (N_samples)
    z_vals = 0.01 * (1 - z_steps) + sample_distance * z_steps
    z_vals = z_vals[None,None,:,None].expand(N_rays,N_directions,corase_sample,1)

    corase_points=points_s+directions_s*z_vals #[N_rays,N_directions,corase_sample,3]

    choose_idx = (masks.squeeze() > 0) #[N_rays_choosed]
    vis=torch.zeros(size=(N_rays,N_directions,1),device=corase_points.device)

    corase_points_choosed=corase_points[choose_idx]
    z_vals_choosed=z_vals[choose_idx]

    N_rays_choosed=z_vals_choosed.shape[0]

    corase_points_choosed=corase_points_choosed.reshape(N_rays_choosed*N_directions,corase_sample,3)
    z_vals_choosed=z_vals_choosed.reshape((N_rays_choosed*N_directions,corase_sample))

    if corase_points_choosed.shape[0]>0:
        _,sample_visibility=inference(fine_sigma,embedding_xyz,corase_points_choosed,z_vals_choosed)
        sample_visibility=sample_visibility.reshape(N_rays_choosed,N_directions,1)
        vis[choose_idx]=sample_visibility
        p_PRT = calcu_PRT.module.computePRT_vis(n=directions_num, order=2, p_vis=vis,phi=phi, theta=theta,
                                                threshold=0.2)  # [N_rays,9]
        print('surface points')
        print(p_PRT.max(),p_PRT.min())
        result = {
            'transport': p_PRT,  # [N_rays,9]
            'sample_visibility': sample_visibility,
            'vis': vis,
            'show':True
        }
    else:
        sample_visibility=vis
        print('vis',vis.shape)
        p_PRT = calcu_PRT.module.computePRT_vis(n=directions_num, order=2, p_vis=vis,phi=phi, theta=theta,
                                                threshold=0.2)  # [N_rays,9]
        print('Empty')
        print(p_PRT.max(),p_PRT.min())
        result = {
            'transport': p_PRT,  # [N_rays,9]
            'sample_visibility': sample_visibility,
            'vis': vis,
            'show':False
        }


    return result

def render_sample_visibility_n(
                fine_sigma,
                embeddings,
                calcu_PRT,
                surface_points=None,#[N,3]
                masks=None,
                directions_num=20,
                noise_std=1,
                chunk=1024 * 32,
                sample_distance=4,
                direction_samples=128,
                near=0.01,
                threshold=0.4
                ):

    def inference(fine_sigma, embedding_xyz, xyz_,z_vals):
        # print('starting infer')
        t0=time.time()
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        # print(N_rays_choosed,N_directions,xyz_.shape)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyzdir_embedded = embedding_xyz(xyz_[i:i + chunk])

            out_chunks += [fine_sigma(xyzdir_embedded, vis_predict=True)]

        out = torch.cat(out_chunks, 0)
        sigmas = out.view(N_rays_choosed*N_directions, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).

        # compute alpha by the formula (3)
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        temp=torch.cumprod(alphas_shifted, -1)
        weights = \
            alphas * temp[:, :-1]  # (N_rays, N_samples_)
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        t1=time.time()
        # print('ending infer time=',t1-t0)
        return weights,temp[:,-1:]

    def computeNormal(depth,h,w):
        new_depth=depth.view(1,1,h,w)
        padding=torch.nn.ReflectionPad2d(1)
        new_depth=padding(new_depth) #[1,1,h+2,w+2]
        depth_pad=new_depth.squeeze()#[h+2,w+2]

        dzdx = (depth_pad[2:2 + h, :] - depth_pad[0:0 + h, :]) / 2.0  # [h,w]
        dzdy = (depth_pad[:, 2:2 + w] - depth_pad[:, 0:0 + w]) / 2.0  # [h,w]
        z = torch.ones_like(depth)
        z= z.cuda()
        normal = torch.cat([dzdx[:, :, None], dzdy[:, :, None], z[:, :, None]], axis=1)
        normal_sum = torch.sqrt(torch.power(dzdx, 2) + torch.power(dzdy, 2) + torch.power(z, 2))
        normal = normal / normal_sum[:, :, None]
        return normal

    def computeNormal_points(points):
        '''
        depth:[h,w,3]
        output:[h,w,3]
        '''
        h, w, _ = points.shape
        padding = torch.nn.ReflectionPad2d(1)
        new_points=points.permute(2,0,1).unsqueeze(0) #[1,3,h,w]
        depth_pad = padding(new_points)
        depth_pad =depth_pad.squeeze(0).permute(1,2,0)

        left_y = depth_pad[2:2 + h, 1:1 + w] - depth_pad[1:1 + h, 1:1 + w]  # [h,w,3]
        left_x = depth_pad[1:1 + h, 0:0 + w] - depth_pad[1:1 + h, 1:1 + w]  # [h,w,3]
        right_x = depth_pad[1:1 + h, 2:2 + w] - depth_pad[1:1 + h, 1:1 + w]
        right_y = depth_pad[0:0 + h, 1:1 + w] - depth_pad[1:1 + h, 1:1 + w]

        face_1 = (torch.cross(right_x.view(-1, 3), right_y.view(-1, 3))).view(h, w, 3)
        face_1 = face_1 / torch.sqrt(torch.sum(torch.power(face_1, 2), dim=2, keepdims=True))

        face_2 = (torch.cross(right_y.view(-1, 3), left_x.view(-1, 3))).view(h, w, 3)
        face_2 = face_2 / torch.sqrt(torch.sum(torch.power(face_2, 2), dim=2, keepdims=True))

        face_3 = (torch.cross(left_x.view(-1, 3), left_y.view(-1, 3))).view(h, w, 3)
        face_3 = face_3 / torch.sqrt(torch.sum(torch.power(face_3, 2), dim=2, keepdims=True))

        face_4 = (torch.cross(left_y.view(-1, 3), right_x.view(-1, 3))).view(h, w, 3)
        face_4 = face_4 / torch.sqrt(torch.sum(torch.power(face_4, 2), dim=2, keepdims=True))

        normal = (face_1 + face_2 + face_3 + face_4) / 4.0
        normal = normal / torch.sqrt(torch.sum(torch.power(normal, 2), dim=2, keepdims=True))
        return normal

    # Extract models from lists
    surface_points=surface_points.detach()
    embedding_xyz = embeddings[0]

    # get directions and embeddings
    directions_,phi, theta=calcu_PRT.module.sampleSphericalDirections_N_rays(surface_points.shape[0],directions_num) #[N,3]

    N_rays, _ = surface_points.shape
    _,N_directions, _ = directions_.shape

    # calculating GT visibility using corase NeRF
    # random choose some rays to calculate GT
    corase_sample=direction_samples
    # fine_sample=64
    points_s=surface_points[:,None,None,:].expand(N_rays,N_directions,corase_sample,3)
    directions_s=directions_[:,:,None,:].expand(N_rays,N_directions,corase_sample,3)
    z_steps = torch.linspace(0, 1, corase_sample, device=surface_points.device)  # (N_samples)
    z_vals = near * (1 - z_steps) + sample_distance * z_steps
    z_vals = z_vals[None,None,:,None].expand(N_rays,N_directions,corase_sample,1)

    corase_points=points_s+directions_s*z_vals #[N_rays,N_directions,corase_sample,3]

    choose_idx = (masks.squeeze() > 0) #[N_rays_choosed]
    vis=torch.zeros(size=(N_rays,N_directions,1),device=corase_points.device)

    corase_points_choosed=corase_points[choose_idx]
    z_vals_choosed=z_vals[choose_idx]

    N_rays_choosed=z_vals_choosed.shape[0]

    corase_points_choosed=corase_points_choosed.reshape(N_rays_choosed*N_directions,corase_sample,3)
    z_vals_choosed=z_vals_choosed.reshape((N_rays_choosed*N_directions,corase_sample))

    if corase_points_choosed.shape[0]>0:
        _,sample_visibility=inference(fine_sigma,embedding_xyz,corase_points_choosed,z_vals_choosed)
        sample_visibility=sample_visibility.reshape(N_rays_choosed,N_directions,1)
        vis[choose_idx]=sample_visibility
        p_PRT = calcu_PRT.module.computePRT_vis(n=directions_num, order=2, p_vis=vis,phi=phi, theta=theta,
                                                threshold=threshold)  # [N_rays,9]
        print('surface points')
        print(p_PRT.max(),p_PRT.min())
        result = {
            'transport': p_PRT,  # [N_rays,9]
            'sample_visibility': sample_visibility,
            'vis': vis,
            'show':True
        }
    else:
        sample_visibility=vis
        print('vis',vis.shape)
        p_PRT = calcu_PRT.module.computePRT_vis(n=directions_num, order=2, p_vis=vis,phi=phi, theta=theta,
                                                threshold=threshold)  # [N_rays,9]
        print('Empty')
        print(p_PRT.max(),p_PRT.min())
        result = {
            'transport': p_PRT,  # [N_rays,9]
            'sample_visibility': sample_visibility,
            'vis': vis,
            'show':False
        }


    return result
