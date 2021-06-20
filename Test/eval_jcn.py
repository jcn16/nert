import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/jcn/桌面/Nerf/nerf_simple/data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def computeNormal(depth):
    '''
    depth:[h,w,1]
    output:[h,w,3]
    '''
    h,w=depth.shape
    depth_pad=np.pad(depth,((1,1),(1,1)),'edge') #[h+2,w+2]
    dzdx=(depth_pad[2:2+h,1:1+w]-depth_pad[0:0+h,1:1+w])/2.0 #[h,w]
    dzdy=(depth_pad[1:1+h,2:2+w]-depth_pad[1:1+h,0:0+w])/2.0 #[h,w]
    z=np.ones_like(depth)
    normal=np.concatenate([-dzdx[:,:,None],-dzdy[:,:,None],z[:,:,None]],axis=2)
    normal_sum=np.sqrt(np.power(dzdx,2)+np.power(dzdy,2)+np.power(z,2))
    normal=normal/normal_sum[:,:,None]
    return normal

def computeNormal_points(points):
    '''
    depth:[h,w,3]
    output:[h,w,3]
    '''
    h,w,_=points.shape
    depth_pad=np.pad(points,((1,1),(1,1),(0,0)),'edge') #[h+2,w+2]

    left_y=depth_pad[2:2+h,1:1+w]-depth_pad[1:1+h,1:1+w] #[h,w,3]
    left_x=depth_pad[1:1+h,0:0+w]-depth_pad[1:1+h,1:1+w] #[h,w,3]
    right_x = depth_pad[1:1 + h, 2:2 + w] - depth_pad[1:1 + h, 1:1 + w]
    right_y=depth_pad[0:0+h,1:1+w]-depth_pad[1:1+h,1:1+w]

    face_1=(np.cross(right_x.reshape((-1,3)),right_y.reshape((-1,3)))).reshape((h,w,3))
    face_1=face_1/np.sqrt(np.sum(np.power(face_1,2),axis=2,keepdims=True))

    face_2 = (np.cross(right_y.reshape((-1, 3)), left_x.reshape((-1, 3)))).reshape((h, w, 3))
    face_2 = face_2 / np.sqrt(np.sum(np.power(face_2, 2), axis=2, keepdims=True))

    face_3 = (np.cross(left_x.reshape((-1, 3)), left_y.reshape((-1, 3)))).reshape((h, w, 3))
    face_3 = face_3 / np.sqrt(np.sum(np.power(face_3, 2), axis=2, keepdims=True))

    face_4 = (np.cross(left_y.reshape((-1, 3)), right_x.reshape((-1, 3)))).reshape((h, w, 3))
    face_4 = face_4 / np.sqrt(np.sum(np.power(face_4, 2), axis=2, keepdims=True))

    normal=(face_1+face_2+face_3+face_4)/4.0
    normal=normal/np.sqrt(np.sum(np.power(normal, 2), axis=2, keepdims=True))
    return normal


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse_sigma = NeRF_sigma()
    nerf_coarse_albedo = NeRF_albedo()
    nerf_fine_sigma = NeRF_sigma()
    nerf_fine_albedo = NeRF_albedo()

    nerf_coarse_sigma = nerf_coarse_sigma.cuda()
    nerf_coarse_albedo = nerf_coarse_albedo.cuda()
    nerf_fine_sigma=nerf_fine_sigma.cuda()
    nerf_fine_albedo=nerf_fine_albedo.cuda()
    nerf_coarse_sigma=torch.nn.DataParallel(nerf_coarse_sigma)
    nerf_coarse_albedo=torch.nn.DataParallel(nerf_coarse_albedo)
    nerf_fine_sigma=torch.nn.DataParallel(nerf_fine_sigma)
    nerf_fine_albedo=torch.nn.DataParallel(nerf_fine_albedo)

    nerf_coarse_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))
    nerf_coarse_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_albedo_23.pt'))
    nerf_fine_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma_23.pt'))
    nerf_fine_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_albedo_23.pt'))

    nerf_coarse_sigma.eval()
    nerf_coarse_albedo.eval()
    nerf_fine_sigma.eval()
    nerf_fine_albedo.eval()

    models = [nerf_coarse_sigma,nerf_coarse_albedo, nerf_fine_sigma,nerf_fine_albedo]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back)

        # normal
        depth_values=results['surface_points'].view(h,w,3).cpu().numpy() #[H,W,3]
        normal=computeNormal_points(depth_values)
        normal=0.5*normal+0.5
        # print(normal[300:500,300:500])
        normal_pred=(normal*255).astype(np.uint8)
        imageio.imwrite(os.path.join(dir_name, f'normal_{i:03d}.png'), normal_pred)

        #visibility
        vis=results['visibility'].view(h,w,1).cpu().numpy() #[H,W,3]
        vis_pred = (vis * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(dir_name, f'vis_{i:03d}.png'), vis_pred)
        # print(vis.shape)
        # print(vis.max(),vis.min())

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        
        if args.save_depth:
            depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
            depth_pred = np.nan_to_num(depth_pred)
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(f'depth_{i:03d}', 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.gif'), imgs, fps=30)
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')