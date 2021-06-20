import torch
#from torchsearchsorted import searchsorted
from .prt_util import *
from .sh_util import *
import random

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


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
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

    def inference(model_sigma,model_albedo, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):

        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded

            if not weights_only:
                sigma,features=model_sigma(xyz_embedded)
                rgbs=model_albedo(dir_embedded[i:i+chunk],features)
                out_chunks += [torch.cat([rgbs,sigma],dim=1)]
            else:
                sigma,_=model_sigma(xyz_embedded)
                out_chunks+=[sigma]

        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        temp=torch.cumprod(alphas_shifted, -1)
        weights = \
            alphas * temp[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        visibility=temp[:,-1:]
        if weights_only:
            return weights

        # compute final weighted outputs
        # from these fomulation, The Weights are really matters, it records all information, with this weights, we can easily get rgbs, z from it.
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights,sigmas[:,:],visibility


    # Extract models from lists
    model_coarse_sigma = models[0]
    model_coarse_albedo = models[1]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    # for test

    if test_time:
        weights_coarse = \
            inference(model_coarse_sigma,model_coarse_albedo, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, weights_coarse ,sigmas,_= \
            inference(model_coarse_sigma,model_coarse_albedo,embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                 }

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        if len(models)<3:
            model_fine_sigma = models[0]
            model_fine_albedo = models[1]
        else:
            model_fine_sigma = models[2]
            model_fine_albedo = models[3]
        rgb_fine, depth_fine, weights_fine,sigmas,visibility = \
            inference(model_fine_sigma,model_fine_albedo, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        surface_points=rays_o+rays_d*(depth_fine.unsqueeze(1)) #[N_rays,3]

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)
        # result['surface_points']=surface_points
        # result['xyz']=xyz_coarse_sampled
        # result['visibility']=visibility #[N,1]
        # result['sigmas']=sigmas.sum(1)


    return result


def render_rays_albedo(models,
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

    This part adds Visibility module, to predict visibility map

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

    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
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
            xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded,sigma_only=weights_only)]

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
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]  # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        # from these fomulation, The Weights are really matters, it records all information, with this weights, we can easily get rgbs, z from it.
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights, sigmas[:, :]

    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)

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

    if test_time:
        weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, weights_coarse, sigmas = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse, #[B,3]
                  'depth_coarse': depth_coarse.unsqueeze(-1), #[B,1]
                  'opacity_coarse': weights_coarse.sum(1)
                  }

    if N_importance > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
        # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine, sigmas = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        surface_points=rays_o+rays_d*(depth_fine.unsqueeze(1)) #[N_rays,3]

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine.unsqueeze(-1)
        result['opacity_fine'] = weights_fine.sum(1)
        result['surface_points']=surface_points
        # result['xyz'] = xyz_coarse_sampled
        result['sigmas'] = sigmas.sum(1)

        # for debug
        # def save_obj_mesh_numpy(mesh_path, verts):
        #     file = open(mesh_path, 'w')
        #
        #     for v in range(verts.shape[0]):
        #         file.write('v %.4f %.4f %.4f\n' % (verts[v][0], verts[v][1], verts[v][2]))
        #     file.close()
        # temp =result['xyz'].view(-1, 3).detach().cpu().numpy()
        # save_obj_mesh_numpy('visualize.obj', temp)
        # print('saving !!!')

    return result



def render_rays_visibility(models,
                embeddings,
                surface_points=None,#[N,3]
                depth=None, #[N,]
                directions_num=20,
                noise_std=1,
                chunk=1024 * 32,
                test_time=False,
                white_back=True,
                sample_distance=4
                ):

    def inference(model, embedding_xyz, xyz_,z_vals, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
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
            xyzdir_embedded = embedding_xyz(xyz_[i:i + chunk])
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays_choosed*N_directions_choosed, N_samples_)
        else:
            rgbsigma = out.view(N_rays_choosed*N_directions_choosed, N_samples_, 4)
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)

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
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights, temp[:,-1:]

        # compute final weighted outputs
        # from these fomulation, The Weights are really matters, it records all information, with this weights, we can easily get rgbs, z from it.
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights,temp[:,-1:]

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
    model_vis = models[0]
    model_corase=models[1]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    calcu_PRT = PRT()
    # get directions and embeddings
    directions,_,_=calcu_PRT.sampleSphericalDirections(directions_num) #[N,3]
    directions_=torch.Tensor(directions).cuda()

    # embedings
    points_embedded=embedding_xyz(surface_points) #[N_rays,points_embeddings]
    direction_embedded=embedding_dir(directions_) #[N_directions,direction_embeddings]

    N_rays,points_dim=points_embedded.shape
    N_directions,direction_dim=direction_embedded.shape

    points_embedded=points_embedded[:,None,:].expand((N_rays,N_directions,points_dim)) #[N_rays,N_directions,_]
    direction_embedded=direction_embedded[None,:,:].expand((N_rays,N_directions,direction_dim))

    points_embedded=points_embedded.reshape(-1,points_dim)
    direction_embedded=direction_embedded.reshape(-1,direction_dim)

    # predict visibility from point x , direction w
    out_chunks = []
    for i in range(0, points_embedded.shape[0], chunk):
        # Embed positions by chunk
        out_chunks += [model_vis(points_embedded[i:i+chunk],direction_embedded[i:i+chunk])]
    p_visibility=torch.cat(out_chunks,0)
    p_visibility=p_visibility.view(N_rays,N_directions,1)

    # calculate SH coffs
    p_PRT=calcu_PRT.computePRT_vis(n=directions_num,order=2,p_vis=p_visibility,threshold=0.2) #[N_rays,9]

    # calculating GT visibility using corase NeRF
    # random choose some rays to calculate GT
    corase_sample=64
    # fine_sample=64
    points_s=surface_points[:,None,None,:].expand(N_rays,N_directions,corase_sample,3)
    directions_s=directions_[None,:,None,:].expand(N_rays,N_directions,corase_sample,3)
    z_steps = torch.linspace(0, 1, corase_sample, device=surface_points.device)  # (N_samples)
    z_vals = 0 * (1 - z_steps) + sample_distance * z_steps
    z_vals = z_vals[None,None,:,None].expand(N_rays,N_directions,corase_sample,1)

    corase_points=points_s+directions_s*z_vals #[N_rays,N_directions,corase_sample,3]

    # random choose rays and directions to supervise
    rays_idx=random.sample(range(N_rays),int(0.5*N_rays))
    directions_idx=random.sample(range(N_directions),int(0.2*N_directions))

    # choosed points Through corase network
    choosed_points=corase_points[rays_idx,:,:,:]
    choosed_points=choosed_points[:,directions_idx,:,:]
    N_rays_choosed, N_directions_choosed,_,_=choosed_points.shape
    choosed_points=choosed_points.reshape(N_rays_choosed*N_directions_choosed,corase_sample,3)

    choosed_z_vals=z_vals[rays_idx,:,:,:]
    choosed_z_vals=choosed_z_vals[:,directions_idx,:,:]
    choosed_z_vals=choosed_z_vals.reshape((N_rays_choosed*N_directions_choosed,corase_sample))

    _,sample_visibility=inference(model_corase,embedding_xyz,choosed_points,choosed_z_vals,weights_only=True)
    sample_visibility=sample_visibility.reshape(N_rays_choosed,N_directions_choosed,1)
    p_visibility=p_visibility[rays_idx,:,:]
    p_visibility=p_visibility[:,directions_idx,:]

    result={
        'sample_vis':sample_visibility,
        'p_vis':p_visibility, #[N_rays_choosed,N_directions_choosed,1]
        'transport':p_PRT #[N_rays,9]
    }


    return result
