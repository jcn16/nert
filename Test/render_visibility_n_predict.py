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
val_dataset = dataset(split='val', **kwargs)
val_data_loader=DataLoader(val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

val_iter=iter(val_data_loader)
# Loading Loss
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
    vis_Nerf.load_state_dict(torch.load('./checkpoints/checkpoints_visibility_smooth/vis_0.pt'))
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
    vis_Nerf.eval()
    corase_Nerf_sigma.eval()
    corase_Nerf_sigma.eval()

    for val_batch in val_data_loader:

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
        chunk=1024,near, threshold
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
                            val_dataset.white_back
                            )
            rendered_ray_vis = \
                render_rays_visibility_test(
                                        vis_Nerf,
                                        fine_Nerf_sigma,
                                        embeddings,
                                        calcu_PRT,
                                        surface_points=rendered_ray_chunks['surface_points_threshold'],#[N,3]
                                        masks=masks[i:i+hparams.chunk],
                                        directions_num=20,
                                        noise_std=1,
                                        chunk_large=hparams.chunk_large,
                                        sample_distance=4,
                                        direction_samples=hparams.direction_samples,
                                        near=hparams.near,
                                        threshold=hparams.threshold
                                        )
            results['p_transport']+=[rendered_ray_vis['p_transport'].detach().cpu()]
            results['gt_transport']+=[rendered_ray_vis['gt_transport'].detach().cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        p_PRT=results['p_transport']
        gt_PRT=results['gt_transport']
        W, H = hparams.img_wh
        p_transport = p_PRT.view(H, W, 9)
        gt_transport = gt_PRT.view(H, W, 9)
        rgbs=rgbs.view(H,W,3)

        # save images
        rgbs=rgbs.cpu().numpy()
        rgbs=np.asarray(rgbs*255,dtype=np.uint8)
        cv2.imwrite('/media/jcn/新加卷/Nerf/Nerf/nerf_my/Test/transport/rgbs.png',rgbs)

        p_transport=p_transport.numpy() #[H,W,9]
        np.save('/media/jcn/新加卷/Nerf/Nerf/nerf_my/Test/transport/transport.npy', p_transport)
        for i in range(9):
            temp=np.asarray(np.clip((0.5*(np.repeat(p_transport[:,:,i:i+1],3,axis=2))+0.5)*255.0,0,255),dtype=np.uint8)
            cv2.imwrite(f'/media/jcn/新加卷/Nerf/Nerf/nerf_my/Test/transport/p_transport_{i}.png',temp)

        gt_transport = gt_transport.numpy()  # [H,W,9]
        np.save('/media/jcn/新加卷/Nerf/Nerf/nerf_my/Test/transport/gt_transport.npy', p_transport)
        for i in range(9):
            temp = np.asarray(np.clip((0.5 * (np.repeat(gt_transport[:, :, i:i + 1], 3, axis=2)) + 0.5) * 255.0, 0, 255),
                              dtype=np.uint8)
            cv2.imwrite(f'/media/jcn/新加卷/Nerf/Nerf/nerf_my/Test/transport/gt_transport_{i}.png', temp)










