import os
import math
import numpy as np
from skimage import io
import torch

from vcnerf.utils import get_root_logger
from .utils.nex_llff_loader import OrbiterDataset
from .builder import DATASETS


def get_rays(h, w, px, py, fx, fy, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h))  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)
    j = j.t().to(device)
    dirs = torch.stack([(i-px)/fx, -(j-py)/fy, -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = torch.tensor(c2w[:3,-1]).expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, fx, fy, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*fx)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*fy)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*fx)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*fy)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


@DATASETS.register_module()
class ShinyDataset(object):
    def __init__(self, 
                 base_dir, 
                 llff_width,
                 batch_size,
                 split,
                 scale=None,
                 cache_size=50,
                 to_cuda=False,
                 batching=False,
                 no_ndc=True,
                 holdout=8,):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)

        img0 = [os.path.join(self.base_dir, 'images', f) 
                for f in sorted(os.listdir(os.path.join(self.base_dir, 'images')))
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = io.imread(img0).shape
        if scale is None:
            scale = float(llff_width / sh[1])

        self.orbiter_dataset = OrbiterDataset(self.base_dir, 
                                              ref_img='', 
                                              scale=scale, 
                                              dmin=-1, 
                                              dmax=-1, 
                                              invz=False, 
                                              render_style='shiny',
                                              cache_size=cache_size,
                                              offset=0,
                                              cv2resize=True)
        all_idx = list(range(len(self.orbiter_dataset)))
        if split == 'train':
            self.valid_idx = list(set(all_idx)-set(all_idx[::holdout]))
        else:
            self.valid_idx = all_idx[::holdout]
        self.logger.info(f'{split} index: {self.valid_idx}')

        item = self.orbiter_dataset[0]
        self.h = item['height']
        self.w = item['width']
        self.near = self.orbiter_dataset.sfm.dmin
        self.far = self.orbiter_dataset.sfm.dmax
        if not no_ndc:
            self.near = 0.
            self.far = 1.
            self.fx = item['fx']
            self.fy = item['fy']
            self.logger.info(f'ndc params: fx {self.fx} fy {self.fy}')
        self.batch_size = batch_size
        self.to_cuda = to_cuda
        self.batching = batching
        self.no_ndc = no_ndc
        self.load_all = cache_size >= len(self.valid_idx)
        if self.load_all:
            self.all_item = []
            for i in self.valid_idx:
                item = self.orbiter_dataset[i]
                item['ori_pose'] = torch.tensor(item['ori_pose'])
                item['image'] = torch.tensor(item['image']).permute([1,2,0])
                if self.to_cuda:
                    item['ori_pose'].cuda()
                    item['image'].cuda()
                self.all_item.append(item)
        self.length = len(self.valid_idx)
        if self.batching:
            assert self.load_all, 'loading all images are necessary for batching'
            all_rays_ori, all_rays_dir, all_rays_color = [], [], []
            self.logger.info(f'creating all rays')
            for item in self.all_item:
                ro, rd = get_rays(self.h, self.w, 
                                  item['px'], item['py'], 
                                  item['fx'], item['fy'],
                                  item['ori_pose'],)
                all_rays_ori.append(ro)
                all_rays_dir.append(rd)
                all_rays_color.append(item['image'])
            self.logger.info(f'finish creating all rays')
            self.all_rays_ori = torch.stack(all_rays_ori, dim=0).view([-1,3])
            self.all_rays_dir = torch.stack(all_rays_dir, dim=0).view([-1,3])
            self.all_rays_color = torch.stack(all_rays_color, dim=0).view([-1,3])
            if not self.no_ndc:
                self.all_rays_ori, self.all_rays_dir = ndc_rays(
                    self.h, self.w, self.fx, self.fy, 1., 
                    self.all_rays_ori, self.all_rays_dir)
            self.length = self.all_rays_color.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.load_all:
            idx = self.valid_idx[idx]
            item = self.orbiter_dataset[idx]
            item['image'] = torch.tensor(item['image']).permute([1,2,0])
        else:
            if self.batching:
                return {'rays_ori': self.all_rays_ori[idx],
                        'rays_dir': self.all_rays_dir[idx],
                        'rays_color': self.all_rays_color[idx],
                        'near': self.near, 'far': self.far}
            item = self.all_item[idx]
        rays_ori, rays_dir = get_rays(self.h, self.w, 
                                        item['px'], item['py'], 
                                        item['fx'], item['fy'],
                                        item['ori_pose'],)
        if not self.no_ndc:
            rays_ori, rays_dir = ndc_rays(
                self.h, self.w, self.fx, self.fy, 1., rays_ori, rays_dir)
        rays_color = item['image']

        if self.batch_size == -1:
            return {'rays_ori': rays_ori.view([-1,3]),
                    'rays_dir': rays_dir.view([-1,3]),
                    'rays_color': rays_color.view([-1,3]),
                    'near': self.near, 'far': self.far}

        select_idx = torch.randperm(self.h*self.w)[:self.batch_size]
        rays_ori = rays_ori.view([-1,3])[select_idx]  # (N, 3)
        rays_dir = rays_dir.view([-1,3])[select_idx]  # (N, 3)
        rays_color = rays_color.view([-1,3])[select_idx]  # (N, 3)

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'near': self.near, 'far': self.far}

