import os, sys
from opt import get_opts
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_SmoothAlbedo,NeRF_SmoothTransport

# metrics
from metrics import *
import time

from termcolor import colored
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

'''
Given pretrained transport map, Train albedo only
'''

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
l1_loss=torch.nn.L1Loss(reduction='mean').cuda()

#Loading Network
print(colored('Loading Network','red'))
Trans_Nerf=NeRF_SmoothTransport().cuda()
Trans_Nerf=torch.nn.DataParallel(Trans_Nerf)

Albedo_Nerf=NeRF_SmoothAlbedo().cuda()
Albedo_Nerf=torch.nn.DataParallel(Albedo_Nerf)

try:
    Trans_Nerf.load_state_dict(torch.load('./checkpoints/checkpoints_transport_0.0001/vis_198.pt'))
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
                "params": Albedo_Nerf.parameters(),
                "lr": 1e-4,
            }
        ])

#Loading Checkpoints
if not os.path.exists(hparams.check_dir):
    os.makedirs(hparams.check_dir)
logger=SummaryWriter(os.path.join(hparams.check_dir, 'logs'))
checkpoints_io=CheckpointIO(hparams.check_dir,
                            Albedo_Nerf=Albedo_Nerf,
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
        Albedo_Nerf.train()
        Trans_Nerf.eval()

        rays=batch['rays'].cuda()
        rgbs=batch['rgbs'].cuda()
        surface_points = batch['surfaces'].cuda()
        masks=batch['masks'].cuda()

        embeddings = [embedding_xyz, embedding_dir]

        def inference(TransNet, embedding_xyz, xyz_,out_channel=3):

            N_rays = xyz_.shape[0]
            # Embed directions
            xyz_ = xyz_.view(-1, 3)
            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            out_chunks = []

            for i in range(0, B, hparams.chunk):
                # Embed positions by chunk
                xyz_embedded = embedding_xyz(xyz_[i:i + hparams.chunk])
                p_sigma = TransNet(xyz_embedded)
                out_chunks += [p_sigma]

            out = torch.cat(out_chunks, 0)
            Trans = out.view(N_rays, out_channel)
            return Trans

        with torch.no_grad():
            p_PRT=inference(Trans_Nerf,embedding_xyz,surface_points,out_channel=9)
        p_albedo=inference(Albedo_Nerf,embedding_xyz,surface_points,out_channel=3)

        xyz_noise = torch.normal(mean=0, std=hparams.normal_thred, size=surface_points.shape,
                                 device=surface_points.device)
        assert xyz_noise.shape == surface_points.shape, "Noise generating is wrong!"
        surface_points_disturb = surface_points + xyz_noise

        p_albedo_disturb=inference(Albedo_Nerf,embedding_xyz,surface_points_disturb,out_channel=3)
        p_shading=p_PRT@(Albedo_Nerf.module.sh)
        p_rgbs = p_albedo * p_shading[:, None]/2

        rgb_loss = mse_loss(p_rgbs, rgbs)
        rgb_smooth_loss=l1_loss(p_albedo,p_albedo_disturb)
        sh=Albedo_Nerf.module.sh
        # regular_loss=torch.sum(torch.pow(sh[1:-1],2))+torch.pow(sh[0]-1,2)
        loss = rgb_loss + rgb_smooth_loss

        psnr_rgb=psnr(p_rgbs,rgbs)
        psnr_rgb=psnr_rgb.mean()

        o_shared.zero_grad()
        loss.backward()
        o_shared.step()

        t3=time.time()

        logger.add_scalar('train/loss', loss, it)
        logger.add_scalar('train/rgb_loss', rgb_loss, it)
        logger.add_scalar('train/rgb_smooth_loss', rgb_smooth_loss, it)
        logger.add_scalar('train/psnr_rgb', psnr_rgb, it)

        # print('Loss=',loss.item(),'rgb_loss=',rgb_loss.item(),'vis_loss=',(vis_loss_outside+vis_loss_inside).item(),'psnr=',psnr_rgb.item())
        # print('data=',t1-t0,'sigma=',t2-t1,'vis=',t3-t2)

        t0=t3
        if it%hparams.step==0:
            print("save models")
            base_path = hparams.check_dir
            model_path = os.path.join(base_path, 'NeRF_SmoothAlbedo_%d.pt' % epoch)
            torch.save(Albedo_Nerf.state_dict(), model_path)
    try:
        val_batch = val_iter.__next__()
    except StopIteration:
        val_iter = iter(val_data_loader)
        val_batch = val_iter.__next__()

    with torch.no_grad():
        Albedo_Nerf.eval()
        Trans_Nerf.eval()

        rays = val_batch['rays'].cuda()
        rays =rays.squeeze()
        rgbs = val_batch['rgbs'].cuda()
        rgbs=rgbs.squeeze()
        masks = val_batch['masks'].cuda()
        masks=masks.squeeze()
        surface_points = val_batch['surfaces'].cuda()
        surface_points=surface_points.squeeze()
        B,_=rays.shape

        embeddings = [embedding_xyz, embedding_dir]

        results = {}

        def inference(TransNet, embedding_xyz, xyz_, out_channel=3):

            N_rays = xyz_.shape[0]
            # Embed directions
            xyz_ = xyz_.view(-1, 3)
            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            out_chunks = []

            for i in range(0, B, hparams.chunk):
                # Embed positions by chunk
                xyz_embedded = embedding_xyz(xyz_[i:i + hparams.chunk])
                p_sigma = TransNet(xyz_embedded)
                out_chunks += [p_sigma]

            out = torch.cat(out_chunks, 0)
            Trans = out.view(N_rays, out_channel)
            return Trans

        for i in range(0, B, hparams.chunk):
            print('starting')
            p_PRT = inference(Trans_Nerf, embedding_xyz, surface_points, out_channel=9)
            p_albedo = inference(Albedo_Nerf, embedding_xyz, surface_points, out_channel=3)

            results['transport']+=[p_PRT]
            results['albedo']+=[p_albedo]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        p_PRT = results['transport']
        p_albedo = results['albedo']
        # print('sh',fine_Nerf_albedo.module.sh)
        p_shading = p_PRT @ (Albedo_Nerf.module.sh)
        p_shading = p_shading / 2
        p_rgbs = p_albedo * p_shading[:, None]

        rgb_loss = mse_loss(p_rgbs, rgbs)

        psnr_rgb = psnr(p_rgbs, rgbs)
        psnr_rgb = psnr_rgb.mean()

        W, H = hparams.img_wh
        img = p_rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
        albedo = p_albedo.view(H, W, 3).permute(2, 0, 1).cpu()
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)

        show_img = torch.cat([img_gt,img,albedo], 2)
        logger.add_image('val/show_images', show_img.byte(), epoch)

        logger.add_scalar('val/loss', rgb_loss, it)
        logger.add_scalar('val/psnr_rgb', psnr_rgb, it)

    print("save models")
    base_path = hparams.check_dir
    model_path = os.path.join(base_path, 'NeRF_SmoothAlbedo_%d.pt' % epoch)
    torch.save(Albedo_Nerf.state_dict(), model_path)



