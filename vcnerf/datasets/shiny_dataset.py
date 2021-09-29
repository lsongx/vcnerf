import os
import numpy as np
from skimage import io
import torch

from vcnerf.utils import get_root_logger
from .utils.nex_llff_loader import OrbiterDataset
from .builder import DATASETS


def get_rays(h, w, px, py, fx, fy, r, t):
    i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-px)/fx, -(j-py)/fy, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * r, -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = torch.tensor(t).view([-1]).expand(rays_d.shape)
    return rays_o, rays_d


@DATASETS.register_module()
class ShinyDataset(object):
    def __init__(self, 
                 base_dir, 
                 llff_width,
                 batch_size,
                 split,
                 cache_size=50,
                 holdout=8,):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)

        img0 = [os.path.join(self.base_dir, 'images', f) 
                for f in sorted(os.listdir(os.path.join(self.base_dir, 'images')))
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = io.imread(img0).shape
        scale = float(llff_width / sh[1])

        self.orbiter_dataset = OrbiterDataset(self.base_dir, 
                                              ref_img='', 
                                              scale=scale, 
                                              dmin=-1, 
                                              dmax=-1, 
                                              invz=False, 
                                              render_style='shiny',
                                              cache_size=cache_size,
                                              offset=0,)
        all_idx = list(range(len(self.orbiter_dataset)))
        if split == 'train':
            self.valid_idx = list(set(all_idx)-set(all_idx[::holdout]))
        else:
            self.valid_idx = all_idx[::holdout]
        self.logger.info(f'{split} index: {self.valid_idx}')

        self.h = self.orbiter_dataset[0]['height']
        self.w = self.orbiter_dataset[0]['width']
        self.near = self.orbiter_dataset.sfm.dmin
        self.far = self.orbiter_dataset.sfm.dmax
        self.batch_size = batch_size

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        idx = self.valid_idx[idx]
        item = self.orbiter_dataset[idx]

        rays_ori, rays_dir = get_rays(self.h, self.w, 
                                      item['px'], item['py'], 
                                      item['fx'], item['fy'],
                                      item['r'], item['t'],)
        rays_color = torch.tensor(item['image']).permute([1,2,0])
        
        if self.batch_size == -1:
            return {'rays_ori': rays_ori.view([-1,3]),
                    'rays_dir': rays_dir.view([-1,3]),
                    'rays_color': rays_color.view([-1,3]),
                    'near': self.near, 'far': self.far}

        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, self.h-1, self.h), 
            torch.linspace(0, self.w-1, self.w)), -1)

        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[self.batch_size], replace=False)  # (N,)
        select_coords = coords[select_inds].long()  # (N, 2)
        rays_ori = rays_ori[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        rays_dir = rays_dir[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        rays_color = rays_color[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'near': self.near, 'far': self.far}

