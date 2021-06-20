import os, sys
from opt import get_opts
from tqdm import tqdm
import time

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_SmoothNormal

# metrics
from metrics import *
from termcolor import colored
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
os.environ['CUDA_VISIBLE_DEVICES']='0'

'''
Using pretrained normal and origin normal as GT

hparams.normal_thredshold=0.0001

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
mse_loss=torch.nn.MSELoss(reduction='sum').cuda()
l1_loss=torch.nn.L1Loss(reduction='sum').cuda()

#Loading Network
print(colored('Loading Network','red'))
Pretrained_NormalNet=NeRF_SmoothNormal().cuda()
Pretrained_NormalNet=torch.nn.DataParallel(Pretrained_NormalNet)

NormalNet=NeRF_SmoothNormal().cuda()
NormalNet=torch.nn.DataParallel(NormalNet)

try:
    Pretrained_NormalNet.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_normal_0.0001/vis_38.pt'))
    print('Loading sigma ...')
except:
    print('Start New Training ..')
    # sys.exit(1)

embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)

o_shared=torch.optim.Adam([
            {
                "params": NormalNet.parameters(),
                "lr": 1e-4,
            }
        ])

#Loading Checkpoints
if not os.path.exists(hparams.check_dir):
    os.makedirs(hparams.check_dir)
logger=SummaryWriter(os.path.join(hparams.check_dir, 'logs'))
checkpoints_io=CheckpointIO(hparams.check_dir,
                            NormalNet=NormalNet,
                            optimizer=o_shared)
try:
    load_dict=checkpoints_io.load('model_latest.pt')
except FileExistsError:
    load_dict=dict()
start_epoch=load_dict.get('epoch_it',0)
it=load_dict.get('it',0)

#Start Training
print(colored('Start Training','blue'))
print("bacth_size=",hparams.batch_size)
pbar=tqdm(total=10000)

for epoch in range(10000):
    NormalNet.train()
    Pretrained_NormalNet.eval()

    epoch=epoch+start_epoch
    print('epoch=',epoch)
    pbar.update(1)

    #all losse
    t0=time.time()
    sub_pbar=tqdm(total=int(len(train_dataset)/hparams.batch_size))

    for batch in train_data_loader:
        it += 1
        sub_pbar.update(1)

        rays=batch['rays'].cuda()
        # gt_transports=batch['transports'][0].cuda()
        gt_normals=batch['normals'].cuda()
        surface_points=batch['surfaces'].cuda()
        masks=batch['masks'].cuda()


        def inference(NormalNet, embedding_xyz, xyz_):

            N_rays = xyz_.shape[0]
            # Embed directions
            xyz_ = xyz_.view(-1, 3)
            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            out_chunks = []

            for i in range(0, B, hparams.chunk):
                # Embed positions by chunk
                xyz_embedded = embedding_xyz(xyz_[i:i + hparams.chunk])
                p_sigma = NormalNet(xyz_embedded)
                out_chunks += [p_sigma]

            out = torch.cat(out_chunks, 0)
            Normals = out.view(N_rays, 3)
            return Normals

        xyz_noise = torch.normal(mean=0, std=hparams.normal_thred, size=surface_points.shape, device=surface_points.device)
        assert xyz_noise.shape == surface_points.shape, "Noise generating is wrong!"
        surface_points_disturb = surface_points + xyz_noise

        p_normal=inference(NormalNet,embedding_xyz,surface_points)
        p_normal_disturb=inference(NormalNet,embedding_xyz,surface_points_disturb)

        with torch.no_grad():
            gt_normal_pretrained=inference(Pretrained_NormalNet,embedding_xyz,surface_points)

        # Compute loss
        # MSE_Loss=mse_loss(p_normal,gt_normals)
        MSE_Loss_Pretrained=mse_loss(p_normal,gt_normal_pretrained)
        Smooth_Loss=l1_loss(p_normal_disturb,p_normal)
        loss=Smooth_Loss+MSE_Loss_Pretrained

        o_shared.zero_grad()
        loss.backward()
        o_shared.step()

        logger.add_scalar('train/Loss', loss, it)
        # logger.add_scalar('train/mse_loss', MSE_Loss, it)
        logger.add_scalar('train/mse_pretrained_loss', MSE_Loss_Pretrained, it)
        logger.add_scalar('train/smooth_loss', Smooth_Loss, it)

    with torch.no_grad():
        NormalNet.eval()
        Pretrained_NormalNet.eval()
        val_batch=val_iter.__next__()
        rays = val_batch['rays'][0].cuda()
        # gt_transports=batch['transports'][0].cuda()
        gt_normals = val_batch['normals'][0].cuda()
        surface_points = val_batch['surfaces'][0].cuda()
        masks = val_batch['masks'][0].cuda()

        def inference(NormalNet, embedding_xyz, xyz_):

            N_rays = xyz_.shape[0]
            # Embed directions
            xyz_ = xyz_.view(-1, 3)
            # Perform model inference to get rgb and raw sigma
            B = xyz_.shape[0]
            out_chunks = []

            for i in range(0, B, hparams.chunk):
                # Embed positions by chunk
                xyz_embedded = embedding_xyz(xyz_[i:i + hparams.chunk])
                p_sigma = NormalNet(xyz_embedded)
                out_chunks += [p_sigma]

            out = torch.cat(out_chunks, 0)
            Normals = out.view(N_rays, 3)
            return Normals

        Normals=inference(NormalNet,embedding_xyz,surface_points)
        Normals=Normals.view(800,800,3).permute(2,0,1)*255.0
        gt_normals=gt_normals.view(800,800,3).permute(2,0,1)*255.0

        with torch.no_grad():
            gt_normal_pretrained=inference(Pretrained_NormalNet,embedding_xyz,surface_points)
            gt_normal_pretrained = gt_normal_pretrained.view(800, 800, 3).permute(2, 0, 1) * 255.0

        show_image=torch.cat([Normals,gt_normal_pretrained,gt_normals],dim=1)
        logger.add_image('val/normal', show_image.byte(), epoch)

    print("save models")
    base_path = hparams.check_dir
    model_path = os.path.join(base_path, 'vis_%d.pt' % epoch)
    torch.save(NormalNet.state_dict(), model_path)
    checkpoints_io.save('model_latest.pt', epoch_it=epoch, it=it)



