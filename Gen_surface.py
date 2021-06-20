import os, sys
from opt import get_opts
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from datasets import dataset_dict
# models
from models.nert import Embedding,NeRF_SmoothTransport,NeRF_sigma
from models.rendering_visibility import render_rays

# metrics
from metrics import *
from termcolor import colored
os.environ['CUDA_VISIBLE_DEVICES']='1'

'''
Generate surface points using threshold = 20
'''
hparams = get_opts()

#Loading Dataset
print(colored('Loading Dataset','red'))
dataset = dataset_dict[hparams.dataset_name]
kwargs = {'root_dir': hparams.root_dir,
          'img_wh': tuple(hparams.img_wh)}
train_dataset = dataset(split='train', **kwargs)
train_data_loader=DataLoader(train_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)

val_dataset = dataset(split='val', **kwargs)
val_data_loader=DataLoader(val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)

test_dataset = dataset(split='test', **kwargs)
test_data_loader=DataLoader(test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)
#Loading Network
print(colored('Loading Network','red'))
# TranNet=NeRF_SmoothTransport().cuda()
# TranNet=torch.nn.DataParallel(TranNet)
corase_Nerf_sigma=NeRF_sigma().cuda()
corase_Nerf_sigma=torch.nn.DataParallel(corase_Nerf_sigma)
fine_Nerf_sigma=NeRF_sigma().cuda()
fine_Nerf_sigma=torch.nn.DataParallel(fine_Nerf_sigma)

try:
    corase_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/corase_sigma_23.pt'))
    fine_Nerf_sigma.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints/fine_sigma_23.pt'))
    # TranNet.load_state_dict(torch.load('/home/jcn/桌面/Nerf/nerf_my/checkpoints_visibility_smooth/vis_0.pt'))
    print('Loading sigma ...')
except:
    print('Start New Training ..')
    sys.exit(1)

embedding_xyz = Embedding(3, 10).cuda()  # 10 is the default number
embedding_xyz= torch.nn.DataParallel(embedding_xyz)

#Start Training
print(colored('Start Training','blue'))
print('total data=',len(train_dataset))
iter_number = int(len(train_dataset) / hparams.batch_size)
print('val data per ', iter_number)

embeddings=[embedding_xyz]
pbar=tqdm(total=len(train_dataset))

with torch.no_grad():
    # TranNet.train()
    corase_Nerf_sigma.eval()
    fine_Nerf_sigma.eval()
    DataLoaders=[train_data_loader,val_data_loader]

    for dataloader in DataLoaders:
        for batch in dataloader:
            pbar.update(1)
            rays=batch['rays'][0].cuda()
            masks=batch['masks'][0].cuda()
            path=batch['path'][0]
            B,_=rays.shape

            surface_points=[]
            sub_pbar=tqdm(total=int(B/hparams.chunk))
            for i in range(0,B,hparams.chunk):
                sub_pbar.update(1)
                results=render_rays(corase_Nerf_sigma,fine_Nerf_sigma,
                                    embeddings,rays[i:i+hparams.chunk],
                                    hparams.N_samples,
                                    hparams.use_disp,
                                    hparams.perturb,
                                    hparams.noise_std,
                                    hparams.N_importance,
                                    hparams.chunk_large,  # chunk size is effective in val mode
                                    train_dataset.white_back
                                    )
                surface_points+=[results['surface_points_threshold'].detach().cpu()]
            surface_points=torch.cat(surface_points,dim=0)
            surface_points=surface_points.numpy()

            # save normal
            target_path=path+'_surface.npy'
            print(target_path)
            np.save(target_path,surface_points)



