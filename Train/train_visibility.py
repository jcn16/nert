import os, sys
from opt import get_opts
from tqdm import tqdm
import time

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility
from models.rendering_visibility import render_rays, render_rays_visibility
from models.sh_util_gpu_nert import PRT
from collections import defaultdict

# metrics
from metrics import *
from utils import visualize_depth

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

vis_Nerf=Visibility().cuda()
vis_Nerf=torch.nn.DataParallel(vis_Nerf)


try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma_23.pt'))
    vis_Nerf.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_visibility/vis_0.pt'))
    print('Loading sigma ...')
except:
    print('Loading error ..')
    sys.exit(1)


embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

o_shared=torch.optim.Adam([
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
                            corase_Nerf_sigma=corase_Nerf_sigma,
                            fine_Nerf_sigma=fine_Nerf_sigma,
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
        fine_Nerf_sigma.eval()
        vis_Nerf.train()

        rays=batch['rays'].cuda()
        rgbs=batch['rgbs'].cuda()
        masks=batch['masks'].cuda()

        embeddings = [embedding_xyz, embedding_dir]

        t1=time.time()
        # sigma , albedo, surface_points
        results=render_rays(corase_Nerf_sigma,fine_Nerf_sigma,
                            embeddings,rays,
                            hparams.N_samples,
                            hparams.use_disp,
                            hparams.perturb,
                            hparams.noise_std,
                            hparams.N_importance,
                            hparams.chunk_large,  # chunk size is effective in val mode
                            train_dataset.white_back
                            )
        t2=time.time()
        # visibility
        results_vis=render_rays_visibility( vis_Nerf,
                                            fine_Nerf_sigma,
                                            embeddings,
                                            calcu_PRT,
                                            surface_points=results['surface_points'],#[N,3]
                                            masks=masks,
                                            directions_num=20,
                                            noise_std=1,
                                            chunk=hparams.chunk,
                                            chunk_large=hparams.chunk_large,
                                            sample_distance=4
                                            )

        vis_loss = mse_loss(results_vis['sample_vis'].view(-1), results_vis['p_vis'].view(-1))

        # print(results_vis['sample_vis'].max(),results_vis['sample_vis'].min())
        # print(torch.sum(((results_vis['sample_vis'].view(-1))>0.00001).float()),results_vis['sample_vis'].shape)
        # print(results_vis['p_vis'].max(),results_vis['p_vis'].min())
        t3=time.time()

        o_shared.zero_grad()
        vis_loss.backward()
        o_shared.step()

        t4=time.time()
        # print("Loss=",vis_loss.item(),'data=',t1-t0,'sigma=',t2-t1,'vis=',t3-t2,'backward=',t4-t3)

        t0=t4

        logger.add_scalar('train/loss', vis_loss, it)

        if it%hparams.step==0:
            print("save models")
            base_path = hparams.check_dir
            model_path = os.path.join(base_path, 'corase_sigma_%d.pt' % epoch)
            torch.save(corase_Nerf_sigma.state_dict(), model_path)
            model_path = os.path.join(base_path, 'fine_sigma_%d.pt' % epoch)
            torch.save(fine_Nerf_sigma.state_dict(), model_path)
            model_path = os.path.join(base_path, 'vis_%d.pt' % epoch)
            torch.save(vis_Nerf.state_dict(), model_path)
            checkpoints_io.save('model_latest.pt', epoch_it=epoch, it=it)

    try:
        val_batch = val_iter.__next__()
    except StopIteration:
        val_iter = iter(val_data_loader)
        val_batch = val_iter.__next__()

    with torch.no_grad():
        corase_Nerf_sigma.eval()
        fine_Nerf_sigma.eval()
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
            print('success!')
            rendered_ray_chunks = \
                render_rays(corase_Nerf_sigma, fine_Nerf_sigma,
                            embeddings, rays[i:i+hparams.chunk],
                            hparams.N_samples,
                            hparams.use_disp,
                            hparams.perturb,
                            hparams.noise_std,
                            hparams.N_importance,
                            hparams.chunk_large,  # chunk size is effective in val mode
                            train_dataset.white_back
                            )
            print('vis')
            rendered_ray_vis = \
                render_rays_visibility(vis_Nerf,
                                       fine_Nerf_sigma,
                                       embeddings,
                                       calcu_PRT,
                                       surface_points=rendered_ray_chunks['surface_points'],  # [N,3]
                                       masks=masks,
                                       directions_num=20,
                                       noise_std=1,
                                       chunk=hparams.chunk,
                                       chunk_large=hparams.chunk_large,
                                       sample_distance=4,
                                       status='train'
                                       )
            results['p_vis']+=[rendered_ray_vis['p_vis']]
            results['sample_vis']+=[rendered_ray_vis['sample_vis']]

        vis_loss = mse_loss(results['sample_vis'].view(-1), results['p_vis'].view(-1))
        logger.add_scalar('val/loss', vis_loss, epoch)

    print("save models")
    base_path = hparams.check_dir
    model_path = os.path.join(base_path, 'corase_sigma_%d.pt' % epoch)
    torch.save(corase_Nerf_sigma.state_dict(), model_path)
    model_path = os.path.join(base_path, 'fine_sigma_%d.pt' % epoch)
    torch.save(fine_Nerf_sigma.state_dict(), model_path)
    model_path = os.path.join(base_path, 'vis_%d.pt' % epoch)
    torch.save(vis_Nerf.state_dict(), model_path)
    checkpoints_io.save('model_latest.pt', epoch_it=epoch, it=it)



