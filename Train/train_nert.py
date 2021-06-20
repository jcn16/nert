import os, sys
from opt import get_opts
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility
from models.rendering_nert import render_rays, render_rays_visibility
from models.sh_util_gpu_nert import PRT
from collections import defaultdict

# metrics
from metrics import *
from utils import visualize_depth
import time

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
corase_Nerf_albedo=NeRF_albedo_light().cuda()
corase_Nerf_albedo=torch.nn.DataParallel(corase_Nerf_albedo)
fine_Nerf_sigma=NeRF_sigma().cuda()
fine_Nerf_sigma=torch.nn.DataParallel(fine_Nerf_sigma)
fine_Nerf_albedo=NeRF_albedo_light().cuda()
fine_Nerf_albedo=torch.nn.DataParallel(fine_Nerf_albedo)

vis_Nerf=Visibility().cuda()
vis_Nerf=torch.nn.DataParallel(vis_Nerf)
try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_2/corase_sigma_0.pt'))
    corase_Nerf_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_2/corase_albedo_0.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_2/fine_sigma_0.pt'))
    fine_Nerf_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_2/fine_albedo_0.pt'))
    vis_Nerf.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_nert_2/vis_nerf_0.pt'))
    print('Loading sigma ...')
except:
    print('Loading error ..')
    # sys.exit(1)


embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

o_shared=torch.optim.Adam([
            {
                "params": corase_Nerf_albedo.parameters(),
                "lr": 1e-4,
            },
            {
                "params": fine_Nerf_albedo.parameters(),
                "lr": 1e-4,
            },
            {
                "params": vis_Nerf.parameters(),
                "lr": 1e-4,
            }
        ])

#Loading Checkpoints
if not os.path.exists(hparams.check_dir):
    os.makedirs(hparams.check_dir)
logger=SummaryWriter(os.path.join(hparams.check_dir, 'logs'))
checkpoints_io=CheckpointIO(hparams.check_dir,
                            corase_Nerf_albedo=corase_Nerf_albedo,
                            fine_Nerf_albedo=fine_Nerf_albedo,
                            vis_Nerf=vis_Nerf,
                            optimizer=o_shared)
try:
    load_dict=checkpoints_io.load('model_latest.pt')
except FileExistsError:
    load_dict=dict()
start_epoch=load_dict.get('epoch_it',0)
it=load_dict.get('it',0)

#Start Training
print(colored('Start Training','blue'))
print('total data=',len(train_dataset))
iter_number = int(len(train_dataset) / hparams.batch_size)
print('val data per ', iter_number)

for epoch in range(100):
    epoch=epoch+start_epoch
    pbar = tqdm(total=iter_number)
    #all losse
    print('New Epoch!')
    t0=time.time()
    for batch in train_data_loader:
        pbar.update(1)
        it += 1
        corase_Nerf_sigma.eval()
        corase_Nerf_albedo.train()
        fine_Nerf_sigma.eval()
        fine_Nerf_albedo.train()
        vis_Nerf.train()

        rays=batch['rays'].cuda()
        rgbs=batch['rgbs'].cuda()
        masks=batch['masks'].cuda()

        embeddings = [embedding_xyz, embedding_dir]

        t1=time.time()
        # sigma , albedo, surface_points
        results=render_rays(corase_Nerf_sigma,corase_Nerf_albedo,fine_Nerf_sigma,fine_Nerf_albedo,
                            embeddings,rays,
                            hparams.N_samples,
                            hparams.use_disp,
                            hparams.perturb,
                            hparams.noise_std,
                            hparams.N_importance,
                            hparams.chunk_large,  # chunk size is effective in val mode
                            train_dataset.white_back
                            )
        # visibility
        t2=time.time()
        results_vis=render_rays_visibility( vis_nerf=vis_Nerf,
                                            fine_sigma=fine_Nerf_sigma,
                                            embeddings=embeddings,
                                            calcu_PRT=calcu_PRT,
                                            surface_points=results['surface_points'],#[N,3]
                                            masks=masks,
                                            directions_num=20,
                                            noise_std=1,
                                            sample_points=hparams.direction_samples,
                                            chunk=hparams.chunk_large,
                                            chunk_large=1024*32,
                                            sample_distance=4,
                                            near=hparams.near,
                                            threshold=hparams.threshold)

        p_PRT=results_vis['transport']
        p_albedo=results['rgb_fine']
        # print('sh',fine_Nerf_albedo.module.sh)
        p_shading=p_PRT@(fine_Nerf_albedo.module.sh)
        # p_shading=p_shading/2
        p_rgbs = p_albedo * p_shading[:, None]

        rgb_loss = mse_loss(p_rgbs, rgbs)
        sh=fine_Nerf_albedo.module.sh
        regular_loss=torch.sum(torch.pow(sh[1:-1],2))+torch.pow(sh[0]-1,2)
        # albedo_loss = mse_loss(p_albedo, rgbs)
        vis_loss_outside = mse_loss(torch.tensor(results_vis['sample_outside'].view(-1)>hparams.threshold,dtype=torch.float),
                                    results_vis['p_outside'].view(-1))
        vis_loss_inside = mse_loss(torch.tensor(results_vis['sample_inside'].view(-1)>hparams.threshold,dtype=torch.float),
                                   results_vis['p_inside'].view(-1))
        loss = rgb_loss + vis_loss_outside +vis_loss_inside +0.01*regular_loss

        psnr_rgb=psnr(p_rgbs,rgbs)
        psnr_rgb=psnr_rgb.mean()

        o_shared.zero_grad()
        loss.backward()
        # print('p_PRT', results_vis['transport'].requires_grad)
        # print('p_PRT', results_vis['transport'].grad)
        o_shared.step()

        t3=time.time()

        logger.add_scalar('train/loss', loss, it)
        logger.add_scalar('train/rgb_loss', rgb_loss, it)
        logger.add_scalar('train/regular_loss', 0.01*regular_loss, it)
        logger.add_scalar('train/vis_loss_outside', vis_loss_outside, it)
        logger.add_scalar('train/vis_loss_inside', vis_loss_inside, it)
        logger.add_scalar('train/psnr_rgb', psnr_rgb, it)

        # print('Loss=',loss.item(),'rgb_loss=',rgb_loss.item(),'vis_loss=',(vis_loss_outside+vis_loss_inside).item(),'psnr=',psnr_rgb.item())
        # print('data=',t1-t0,'sigma=',t2-t1,'vis=',t3-t2)

        t0=t3
        if it%hparams.step==0:
            print("save models")
            base_path = hparams.check_dir
            model_path = os.path.join(base_path, 'corase_sigma_%d.pt' % epoch)
            torch.save(corase_Nerf_sigma.state_dict(), model_path)
            model_path = os.path.join(base_path, 'corase_albedo_%d.pt' % epoch)
            torch.save(corase_Nerf_albedo.state_dict(), model_path)
            model_path = os.path.join(base_path, 'fine_sigma_%d.pt' % epoch)
            torch.save(fine_Nerf_sigma.state_dict(), model_path)
            model_path = os.path.join(base_path, 'fine_albedo_%d.pt' % epoch)
            torch.save(fine_Nerf_albedo.state_dict(), model_path)
            model_path = os.path.join(base_path, 'vis_nerf_%d.pt' % epoch)
            torch.save(vis_Nerf.state_dict(), model_path)
    '''
    try:
        val_batch = val_iter.__next__()
    except StopIteration:
        val_iter = iter(val_data_loader)
        val_batch = val_iter.__next__()

    with torch.no_grad():
        corase_Nerf_sigma.eval()
        corase_Nerf_albedo.eval()
        fine_Nerf_sigma.eval()
        fine_Nerf_albedo.eval()
        vis_Nerf.eval()

        rays = val_batch['rays'].cuda()
        rays =rays.squeeze()
        rgbs = val_batch['rgbs'].cuda()
        rgbs=rgbs.squeeze()
        masks = val_batch['masks'].cuda()
        masks=masks.squeeze()
        B,_=rays.shape

        embeddings = [embedding_xyz, embedding_dir]

        results = defaultdict(list)
        for i in range(0, B, hparams.chunk):
            print('starting')
            rendered_ray_chunks = \
                render_rays(corase_Nerf_sigma, corase_Nerf_albedo, fine_Nerf_sigma, fine_Nerf_albedo,
                            embeddings, rays[i:i+hparams.chunk],
                            hparams.N_samples,
                            hparams.use_disp,
                            hparams.perturb,
                            hparams.noise_std,
                            hparams.N_importance,
                            hparams.chunk_large,  # chunk size is effective in val mode
                            train_dataset.white_back
                            )

            rendered_ray_vis = render_rays_visibility(vis_nerf=vis_Nerf,
                                                 fine_sigma=fine_Nerf_sigma,
                                                 embeddings=embeddings,
                                                 calcu_PRT=calcu_PRT,
                                                 surface_points=rendered_ray_chunks['surface_points'],  # [N,3]
                                                 masks=masks[i:i+hparams.chunk],
                                                 directions_num=20,
                                                 noise_std=1,
                                                 sample_points=hparams.direction_samples,
                                                 chunk=hparams.chunk,
                                                 chunk_large=1024 * 32,
                                                 sample_distance=4,
                                                 near=hparams.near,
                                                 threshold=hparams.threshold)

            for k, v in rendered_ray_vis.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        p_PRT = results['transport']
        p_albedo = results['rgb_fine']
        # print('sh',fine_Nerf_albedo.module.sh)
        p_shading = p_PRT @ (fine_Nerf_albedo.module.sh)
        p_shading = p_shading / 2
        p_rgbs = p_albedo * p_shading[:, None]

        rgb_loss = mse_loss(p_rgbs, rgbs)
        # albedo_loss = mse_loss(p_albedo, rgbs)
        vis_loss_outside = mse_loss(
            torch.tensor(results['sample_outside'].view(-1) > hparams.threshold, dtype=torch.float),
            results['p_outside'].view(-1))
        vis_loss_inside = mse_loss(
            torch.tensor(results['sample_inside'].view(-1) > hparams.threshold, dtype=torch.float),
            results['p_inside'].view(-1))
        loss = rgb_loss + vis_loss_outside + vis_loss_inside

        psnr_rgb = psnr(p_rgbs, rgbs)
        psnr_rgb = psnr_rgb.mean()

        W, H = hparams.img_wh
        img = p_rgbs.view(H, W, 3).cpu()
        img = img.permute(2, 0, 1)  # (3, H, W)

        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        depth = visualize_depth(results['depth_fine'].view(H, W))  # (3, H, W)

        show_img = torch.cat([img_gt,img,depth], 2)
        logger.add_image('val/show_images', show_img.byte(), epoch)

        logger.add_scalar('val/loss', loss, it)
        logger.add_scalar('val/rgb_loss', rgb_loss, it)
        logger.add_scalar('val/vis_loss_outside', vis_loss_outside, it)
        logger.add_scalar('val/vis_loss_inside', vis_loss_inside, it)
        logger.add_scalar('val/psnr_rgb', psnr_rgb, it)

    print("save models")
    base_path = hparams.check_dir
    model_path = os.path.join(base_path,'corase_sigma_%d.pt' % epoch)
    torch.save(corase_Nerf_sigma.state_dict(), model_path)
    model_path = os.path.join(base_path, 'corase_albedo_%d.pt' % epoch)
    torch.save(corase_Nerf_albedo.state_dict(), model_path)
    model_path = os.path.join(base_path, 'fine_sigma_%d.pt' % epoch)
    torch.save(fine_Nerf_sigma.state_dict(), model_path)
    model_path = os.path.join(base_path, 'fine_albedo_%d.pt' % epoch)
    torch.save(fine_Nerf_albedo.state_dict(), model_path)
    model_path = os.path.join(base_path, 'vis_nerf_%d.pt' % epoch)
    torch.save(vis_Nerf.state_dict(), model_path)
    '''


