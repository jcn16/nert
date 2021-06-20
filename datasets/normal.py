import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2

from .ray_utils import *

class NormalDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh
        # self.focal=400

        # bounds, common for all scenes
        self.near = 2
        self.far = 6
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)


        if self.split == 'train_': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks=[]
            self.all_surfaces=[]
            self.all_normals=[]

            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)

                if img.shape[0]>3:
                    img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                    mask=(img[:,-1:]>0).float()
                    # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                    img = img[:, :3]*img[:, -1:] # blend A to RGB
                else:
                    img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                    img = img[:, :3]

                normal_path=os.path.join(self.root_dir,f"{frame['file_path']}_normal_surface.png")
                normal = Image.open(normal_path)
                normal = normal.resize(self.img_wh, Image.LANCZOS)
                normal = self.transform(normal)  # (4, h, w)
                normal = normal.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                normal = normal[:, :3]

                self.all_rgbs += [img]
                self.all_normals+=[normal]
                self.all_masks+=[mask]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)

                surface = np.load(os.path.join(self.root_dir,f"{frame['file_path']}_surface.npy"))  # [h*w,3]
                surface=torch.Tensor(surface)
                self.all_surfaces += [surface]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 8)
            self.all_surfaces = torch.cat(self.all_surfaces, 0) # (len(self.meta['frames])*h*w, 9)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_normals = torch.cat(self.all_normals, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_masks = torch.cat(self.all_masks, 0) # (len(self.meta['frames])*h*w, 1)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train_':
            return len(self.all_rays)
        if self.split == 'val':
            return len(self.meta['frames']) # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train_': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'normals': self.all_normals[idx],
                      'surfaces': self.all_surfaces[idx],
                      'masks':self.all_masks[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            _,h,w=img.shape

            if img.shape[0] > 3:
                # print('RGBA')
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                valid_mask = (img[:, -1:] > 0).float()
                img = img[:, :3] * img[:, -1:] # blend A to RGB
            else:
                # print('RGB')
                valid_mask = torch.ones(size=(h,w))  # (H*W) valid color area
                img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3]

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            normal_path = os.path.join(self.root_dir, f"{frame['file_path']}_normal_surface.png")
            normal = Image.open(normal_path)
            normal = normal.resize(self.img_wh, Image.LANCZOS)
            normal = self.transform(normal)  # (4, h, w)
            normal = normal.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            normal = normal[:, :3]

            surface = np.load(os.path.join(self.root_dir, f"{frame['file_path']}_surface.npy"))  # [h*w,3]
            surface = torch.Tensor(surface)

            sample = {
                    'path':os.path.join(self.root_dir, f"{frame['file_path']}"),
                    'rays': rays,
                      'rgbs': img,
                      'normals': normal,
                      'surfaces': surface,
                      'c2w': c2w,
                      'masks': valid_mask}

        return sample