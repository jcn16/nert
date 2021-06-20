import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from opt import get_opts
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import blender_res
# models
from models.nert import Embedding,NeRF_sigma,NeRF_albedo_light,Visibility,Surface_res
from models.rendering_nert_res import render_rays_surface_res
from models.sh_util_gpu_nert import PRT

# metrics
from metrics import *

from termcolor import colored
from tensorboardX import SummaryWriter
from checkpoints import CheckpointIO
os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.set_printoptions(profile="full")

'''
Using pretained network contained in checkpoints_nert_1

This is to train Surface_res network
'''

hparams = get_opts()

#Loading Dataset
print(colored('Loading Dataset','red'))
kwargs = {'root_dir': hparams.root_dir,
          'img_wh': tuple(hparams.img_wh)}
train_dataset = blender_res.BlenderDataset(split='val', **kwargs)
val_dataset = blender_res.BlenderDataset(split='val', **kwargs)

train_data_loader=DataLoader(train_dataset,
                          shuffle=False,
                          num_workers=10,
                          batch_size=1,
                          pin_memory=True)
val_data_loader=DataLoader(val_dataset,
                          shuffle=False,
                          num_workers=10,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

# Loading Loss
mse_loss=torch.nn.MSELoss(reduction='sum').cuda()
calcu_PRT=PRT().cuda()
calcu_PRT=torch.nn.DataParallel(calcu_PRT)

#Loading Network
print(colored('Loading Network','red'))

surface_res=Surface_res().cuda()
surface_res=torch.nn.DataParallel(surface_res)

embedding_xyz = Embedding(3, 4).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)
embedding_dir = Embedding(3, 10).cuda()  # 4 is the default number
embedding_dir=torch.nn.DataParallel(embedding_dir)

o_shared=torch.optim.Adam([
            {
                "params": surface_res.parameters(),
                "lr": 1e-4,
            }
        ])

#Loading Checkpoints
if not os.path.exists(hparams.check_dir):
    os.makedirs(hparams.check_dir)
logger=SummaryWriter(os.path.join(hparams.check_dir, 'logs'))
checkpoints_io=CheckpointIO(hparams.check_dir,
                            Surface_res=surface_res,
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
iter_number = len(train_dataset)
print('val data per ', iter_number)

pbar = tqdm(total=1000)
#all losse
print('New Epoch!')
for epoch in range(100000):
    epoch=epoch+start_epoch
    pbar.update(1)
    for batch in train_data_loader:
        pbar.update(1)
        it += 1

        surface_res.train()

        embeddings = [embedding_xyz, embedding_dir]

        rays=batch['rays'][0].cuda()
        masks=batch['masks'][0].cuda()
        depth=batch['depth'][0].cuda()
        choosed_points=batch['choosed_rays'][0].cuda()
        choosed_depth=batch['choosed_depth'][0].cuda()
        choosed_masks=batch['choosed_masks'][0].cuda()

        # calculate surface points, PRT, surface points
        # visibility
        results_vis=render_rays_surface_res(
            embeddings=embeddings,
            resnet=surface_res,
            all_rays=rays,
            all_depth=depth,
            masks=masks,
            choosed_points=choosed_points,
            choosed_depth=choosed_depth,
            choosed_masks=choosed_masks
        )
        loss_inside=mse_loss(results_vis['p_depth_inside'],results_vis['gt_depth_inside'])
        loss_outside=mse_loss(results_vis['p_outside_res'],results_vis['gt_outside_res'])
        loss=loss_inside+loss_outside

        o_shared.zero_grad()
        loss.backward()
        # print('res',results_vis['p_depth_inside'].grad)
        o_shared.step()

        logger.add_scalar('train/loss', loss, it)
        logger.add_scalar('train/inside_loss', loss_inside, it)
        logger.add_scalar('train/outside_loss', loss_outside, it)

        print("save models")
        base_path = hparams.check_dir
        model_path = os.path.join(base_path, 'surface_res.pt')
        torch.save(surface_res.state_dict(), model_path)











