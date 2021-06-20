import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2

from .ray_utils import *

class BlenderDataset(Dataset):
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
            
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks=[]
        self.all_depth=[]
        self.choosed_depth=[]
        self.choosed_rays=[]
        self.choosed_masks=[]
        for frame in self.meta['frames']:
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, H, W)
            _, h, w = img.shape

            if img.shape[0] > 3:
                # print('RGBA')
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                valid_mask = (img[:, -1:] > 0).float()
                img = img[:, :3] * img[:, -1:]  # blend A to RGB
            else:
                # print('RGB')
                valid_mask = torch.ones(size=(h, w))  # (H*W) valid color area
                img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3]

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d,
                              self.near * torch.ones_like(rays_o[:, :1]),
                              self.far * torch.ones_like(rays_o[:, :1])],
                             1)  # (H*W, 8)

            choosed_points_mask=torch.from_numpy(np.load(os.path.join(self.root_dir, f"{frame['file_path']}_choosed.npy")))
            depth=torch.from_numpy(np.load(os.path.join(self.root_dir, f"{frame['file_path']}_depth.npy")))
            depth=depth.squeeze()

            padding=torch.nn.ZeroPad2d(padding=(1,1,1,1))
            depth_p=padding(depth) #[802,802]
            depth_up=depth_p[0:800,1:801]
            depth_down=depth_p[2:802,1:801]
            depth_left=depth_p[1:801,0:800]
            depth_right=depth_p[1:801,2:802]
            mean_depth=(depth_left+depth_right+depth_down+depth_up)/4.0
            mean_depth=mean_depth.view(-1)

            choosed_points_mask=choosed_points_mask.view(-1)
            choosed_rays=rays[choosed_points_mask]
            choosed_depth=mean_depth[choosed_points_mask]


            self.all_rays+=[rays]
            self.all_rgbs+=[img]
            self.all_masks+=[valid_mask]
            self.all_depth+=[depth.view(-1)]
            self.choosed_masks+=[choosed_points_mask]
            self.choosed_rays+=[choosed_rays]
            self.choosed_depth+=[choosed_depth]


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
            # return len(self.all_rays)
            return 1

    def __getitem__(self, idx):
        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx],
                  'masks':self.all_masks[idx],
                  'depth':self.all_depth[idx],
                  'choosed_masks':self.choosed_masks[idx],
                  'choosed_rays': self.choosed_rays[idx],
                  'choosed_depth': self.choosed_depth[idx]
                  }

        return sample