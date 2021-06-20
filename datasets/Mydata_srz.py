import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import random

from .ray_utils import *

class MyDataset_srz(Dataset):
    def __init__(self, root_dir, split='train', data_name=None,img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        self.data_name=data_name

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,'json',self.data_name,
                               f"{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh

        # bounds, common for all scenes
        self.near = 1
        self.far = 10
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)

            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for key in self.meta.keys():
                pose = np.array(self.meta[key]['extric'])
                # pose=np.linalg.inv(pose)
                pose=pose[:3,:4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                intric=np.array(self.meta[key]['intric'])

                self.directions = \
                    get_ray_directions_intric(h, w, intric)  # (h, w, 3)


                image_path = os.path.join(self.root_dir, 'img',self.data_name, f"{key}.jpg")
                # mask_path=os.path.join(self.root_dir, 'masks', f"{key}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                # mask=Image.open(mask_path).convert('L')
                img = img.resize(self.img_wh, Image.LANCZOS)
                # mask = mask.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                # mask =self.transform(mask) #(1,h,w)

                #img=img*mask
                if img.shape[0]>3:
                    img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                    img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                else:
                    img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                    img=img[:,0:3]
                # img = img[:,0:3]
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta.keys())-1

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            with open(os.path.join(self.root_dir,'json',self.data_name,
                                   f"{self.split}.json"), 'r') as f:
                self.meta = json.load(f)
            keys=list(self.meta.keys())
            # keys=keys.remove('focal')
            # print(keys)

            idx=random.randint(0,len(keys))
            # print(keys[idx])
            # print(keys)

            pose = np.array(self.meta[keys[idx]]['extric'])
            # pose = np.linalg.inv(pose)
            c2w=torch.Tensor(pose[:3,:4])

            intric = np.array(self.meta[keys[idx]]['intric'])
            w, h = self.img_wh
            self.directions = \
                get_ray_directions_intric(h, w, intric)  # (h, w, 3)

            # print(keys[idx],c2w)

            image_path = os.path.join(self.root_dir, 'img',self.data_name, f"{keys[idx]}.jpg")
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)

            img = self.transform(img)  # (4, h, w)
            # print(mask.shape)
            #img = img * mask
            if img.shape[0]>3:

                valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
                img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            else:
                mask_path = os.path.join(self.root_dir, 'mask', self.data_name, f"{keys[idx]}.jpg")
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(self.img_wh, Image.LANCZOS)
                mask = self.transform(mask)  # (4, h, w)
                valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
                # print(valid_mask.shape)
                img = img.view(3, -1).permute(1, 0)  # (H*W, 4) RGBA
                img = img[:,0:3]
                # print('img',img.shape)

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample