import os
import os.path as osp
import random
import json
import numpy as np
import imageio
import cv2
import torch

from vcnerf.utils import get_root_logger
from .synthetic_dataset import get_rays
from .builder import DATASETS
from .utils.llff_loader import load_llff_data


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


@DATASETS.register_module()
class LLFFDataset:
    def __init__(self, 
                 datadir, 
                 factor, 
                 batch_size, 
                 split, 
                 spherify, 
                 no_ndc, 
                 holdout,):
        self.logger = get_root_logger()
        self.batch_size = batch_size
        self.no_ndc = no_ndc
        datadir = osp.expanduser(datadir)
        images, poses, bds, render_poses, i_test = load_llff_data(
            datadir, factor, recenter=True, bd_factor=.75, spherify=spherify)
        self.hwf = poses[0, :3, -1]
        self.h = int(self.hwf[0])
        self.w = int(self.hwf[1])
        self.focal = self.hwf[2]
        self.poses = poses[:, :3, :4]
        self.logger.info(f'Loaded llff images.shape {images.shape}, '
                    f'render_poses.shape {render_poses.shape}, '
                    f'hwf {self.hwf}, datadir {datadir}')
        if not isinstance(i_test, list):
            i_test = [i_test]

        if holdout > 0:
            self.logger.info(f'Auto LLFF holdout: {holdout}')
            i_test = np.arange(images.shape[0])[::holdout]

        i_val = i_test
        if split == "train":
            all_idx = np.array([
                i for i in np.arange(int(images.shape[0])) 
                if (i not in i_test and i not in i_val)])
        else:
            all_idx = i_val

        self.logger.info('DEFINING BOUNDS')
        if no_ndc:
            self.near = bds.min() * 0.9
            self.far = bds.max() * 1.0
        else:
            self.near = 0.
            self.far = 1.
        self.logger.info(f'NEAR {self.near} FAR {self.far}')

        self.imgs = torch.tensor(images[all_idx])
        self.poses = torch.tensor(self.poses[all_idx])
        self.render_poses = torch.tensor(render_poses)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        target = self.imgs[idx%len(self.imgs)]
        pose = self.poses[idx, :3,:4]
        rays_ori, rays_dir = get_rays(self.h, self.w, self.focal, pose)

        if self.batch_size == -1:
            rays_color = target.view([-1,3])  # (N, 3)
            return {'rays_ori': rays_ori.view([-1,3]), 
                    'rays_dir': rays_dir.view([-1,3]), 
                    'rays_color': rays_color, 
                    'near': self.near, 'far': self.far}

        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, self.h-1, self.h), 
            torch.linspace(0, self.w-1, self.w)), -1)

        coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[self.batch_size], replace=False)  # (N,)
        select_coords = coords[select_inds].long()  # (N, 2)
        rays_ori = rays_ori[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        rays_dir = rays_dir[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)
        rays_color = target[select_coords[:, 0], select_coords[:, 1]]  # (N, 3)

        if not self.no_ndc:
            rays_ori, rays_dir = ndc_rays(
                self.h, self.w, self.focal, 1, rays_ori, rays_dir)

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'near': self.near, 'far': self.far}


