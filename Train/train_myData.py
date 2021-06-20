import os, sys
from opt import get_opts
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nerf import Embedding, NeRF,NeRF_sigma,NeRF_albedo
from models.rendering_part import render_rays
from losses import MSELoss
from collections import defaultdict

# metrics
from metrics import *
from utils import visualize_depth

from termcolor import colored
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
os.environ['CUDA_VISIBLE_DEVICES']='1'

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
mse_loss=MSELoss()

#Loading Network
print(colored('Loading Network','red'))
corase_Nerf_sigma=NeRF_sigma().cuda()
corase_Nerf_sigma=torch.nn.DataParallel(corase_Nerf_sigma)
corase_Nerf_albedo=NeRF_albedo().cuda()
corase_Nerf_albedo=torch.nn.DataParallel(corase_Nerf_albedo)
fine_Nerf_sigma=NeRF_sigma().cuda()
fine_Nerf_sigma=torch.nn.DataParallel(fine_Nerf_sigma)
fine_Nerf_albedo=NeRF_albedo().cuda()
fine_Nerf_albedo=torch.nn.DataParallel(fine_Nerf_albedo)
try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma.pt'))
    corase_Nerf_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_albedo.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma.pt'))
    fine_Nerf_albedo.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_albedo.pt'))
    print('Continue Training ...')
except:
    print('Start New Trainging ..')

embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 4).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

o_shared=torch.optim.Adam([
            {
                "params": corase_Nerf_sigma.parameters(),
                "lr": 5e-4,
            },
            {
                "params": corase_Nerf_albedo.parameters(),
                "lr": 5e-4,
            },
            {
                "params": fine_Nerf_sigma.parameters(),
                "lr": 5e-4,
            },
            {
                "params": fine_Nerf_albedo.parameters(),
                "lr": 5e-4,
            }
        ])

#Loading Checkpoints
if not os.path.exists(hparams.check_dir):
    os.makedirs(hparams.check_dir)
logger=SummaryWriter(os.path.join(hparams.check_dir, 'logs'))
checkpoints_io=CheckpointIO(hparams.check_dir,
                            corase_Nerf_sigma=corase_Nerf_sigma,
                            corase_Nerf_albedo=corase_Nerf_albedo,
                            fine_Nerf_sigma=fine_Nerf_sigma,
                            fine_Nerf_albedo=fine_Nerf_albedo,
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

    for batch in train_data_loader:
        pbar.update(1)
        it += 1
        corase_Nerf_sigma.train()
        corase_Nerf_albedo.train()
        fine_Nerf_sigma.train()
        fine_Nerf_albedo.train()

        rays=batch['rays'].cuda()
        rgbs=batch['rgbs'].cuda()

        embeddings = [embedding_xyz, embedding_dir]

        results=render_rays(corase_Nerf_sigma,corase_Nerf_albedo,fine_Nerf_sigma,fine_Nerf_albedo,
                            embeddings,rays,
                            hparams.N_samples,
                            hparams.use_disp,
                            hparams.perturb,
                            hparams.noise_std,
                            hparams.N_importance,
                            hparams.chunk,  # chunk size is effective in val mode
                            train_dataset.white_back
                            )
        loss=mse_loss(results,rgbs)
        loss=loss.mean()
        psnr_=psnr(results['rgb_fine'],rgbs)
        psnr_=psnr_.mean()

        o_shared.zero_grad()
        loss.backward()
        o_shared.step()

        logger.add_scalar('train/loss', loss, it)
        logger.add_scalar('train/psnr', psnr_, it)

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

                rays = val_batch['rays'].cuda()
                rays =rays.squeeze()
                rgbs = val_batch['rgbs'].cuda()
                rgbs=rgbs.squeeze()
                B,_=rays.shape

                embeddings = [embedding_xyz, embedding_dir]

                results = defaultdict(list)
                for i in range(0, B, hparams.chunk):
                    rendered_ray_chunks = \
                        render_rays(corase_Nerf_sigma, corase_Nerf_albedo, fine_Nerf_sigma, fine_Nerf_albedo,
                                    embeddings, rays[i:i+hparams.chunk],
                                    hparams.N_samples,
                                    hparams.use_disp,
                                    hparams.perturb,
                                    hparams.noise_std,
                                    hparams.N_importance,
                                    hparams.chunk,  # chunk size is effective in val mode
                                    train_dataset.white_back
                                    )
                    for k, v in rendered_ray_chunks.items():
                        results[k] += [v]

                for k, v in results.items():
                    results[k] = torch.cat(v, 0)

                val_loss=mse_loss(results,rgbs)
                val_loss=val_loss.mean()
                psnr_ = psnr(results['rgb_fine'], rgbs)
                psnr_ = psnr_.mean()

                W, H = hparams.img_wh
                img = results['rgb_fine'].view(H, W, 3).cpu()
                img = img.permute(2, 0, 1)  # (3, H, W)
                img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                depth = visualize_depth(results['depth_fine'].view(H, W))  # (3, H, W)

                show_img = torch.cat([img_gt,img,depth], 2)
                show_img=show_img*255
                logger.add_image('val/show_images', show_img.byte(), epoch)

                logger.add_scalar('val/loss',val_loss)
                logger.add_scalar('val/psnr',psnr_)

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


