import os
import glob
import random
import numpy as np
from PIL import Image
import torch

from vcnerf.utils import get_root_logger
from .builder import DATASETS


def uvst2rays(uv, st, plane_dist):
    z0 = torch.zeros_like(uv[...,:1])+0.1
    rays_ori = torch.cat([uv, z0], dim=-1)
    z1 = z0 + plane_dist
    rays_dir = torch.cat([st, z1], dim=-1)
    rays_dir /= rays_dir.norm(dim=-1,keepdim=True)
    return rays_ori, rays_dir


@DATASETS.register_module()
class StanfordLFDataset(object):
    def __init__(self, 
                 base_dir, 
                 downsample,
                 batch_size,
                 split,
                 pixel_move,
                 grid,
                 testskip=8,
                 keep_idx=[],
                 scale=1024,
                 perturb=False,
                 to_cuda=True):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)

        all_uv = []
        all_st = []
        all_color = []
        all_name  = []
        # image_paths = glob.glob(f"{self.base_dir}/sparse/*.png")
        image_paths = glob.glob(f"{self.base_dir}/*.png")
        image_paths.sort()
        if keep_idx:
            if isinstance(keep_idx[0], int):
                sel = lambda x: x in keep_idx
            elif isinstance(keep_idx[0], str):
                all_idx = []
                for idx, i in enumerate(image_paths):
                    for keep in keep_idx:
                        if keep in i:
                            all_idx.append(idx)
                            break
                sel = lambda x: x in all_idx
        else:
            sel = (lambda x: x%testskip!=0) if split=='train' else (lambda x: x%testskip==0)
        image_paths = [i for idx, i in enumerate(image_paths) if sel(idx)]

        for p in image_paths:
            all_name.append(p.split('/')[-1])
            img = Image.open(p)
            w, h = img.size
            img = img.resize([w//downsample, h//downsample])
            all_color.append(np.asarray(img)/255.0)
            uv = np.asarray([p.split('_')[3], p.split('_')[4].replace('.png','')], dtype='float32')
            xs = np.arange(0, h//downsample)
            ys = np.arange(0, w//downsample)
            s, t = np.meshgrid(xs, ys, indexing="ij")
            st = np.stack([s, t], -1)*downsample+uv
            self.st_base = np.stack([s, t], -1)*downsample
            all_uv.append(uv)
            all_st.append(st)
        self.w, self.h = w//downsample, h//downsample
        self.downsample = downsample
        self.batch_size = batch_size
        self.scale = scale

        self.all_uv = torch.from_numpy(np.stack(all_uv, axis=0)).float()
        self.all_st = torch.from_numpy(np.stack(all_st, axis=0)).float()
        self.all_color = torch.from_numpy(np.stack(all_color, axis=0)).float()
        self.all_name = all_name
        self.st_base = torch.from_numpy(self.st_base).float()

        self.u_max, self.v_max = self.all_uv.view([-1,2]).max(0).values
        self.u_min, self.v_min = self.all_uv.view([-1,2]).min(0).values
        self.s_max, self.t_max = self.all_st.view([-1,2]).max(0).values
        self.s_min, self.t_min = self.all_st.view([-1,2]).min(0).values

        fov = 57 * np.pi / 180
        self.plane_dist = 0.5 * w / np.tan(0.5 * fov)
        u_change = (self.u_max-self.u_min)/grid
        s_change_min = u_change+pixel_move[0]
        s_change_max = u_change+pixel_move[1]
        self.near = self.plane_dist / (1-u_change/s_change_min)
        self.far = self.plane_dist / (1-u_change/s_change_max)
        import pdb;pdb.set_trace()

        if to_cuda:
            self.all_uv = self.all_uv.cuda()
            self.all_st = self.all_st.cuda()
            self.all_color = self.all_color.cuda()
            self.st_base = self.st_base.cuda()
        
        if perturb and not to_cuda:
            raise RuntimeError('perturb requires to cuda')
        self.perturb = perturb

    def __len__(self):
        return self.all_uv.shape[0]

    def __getitem__(self, idx):
        st = self.all_st[idx]
        rays_color = self.all_color[idx]

        if self.perturb:
            new_st = st + torch.rand_like(st)-0.5
            st_min = st.view([-1,2]).min(dim=0).values
            st_max = st.view([-1,2]).max(dim=0).values
            new_st[:,:,0] = new_st[:,:,0].clamp(st_min[0], st_max[0])
            new_st[:,:,1] = new_st[:,:,1].clamp(st_min[1], st_max[1])
            new_st_grid_coord = ((new_st-st_min)/(st_max-st_min)-0.5)*2
            new_color = torch.nn.functional.grid_sample(
                rays_color.permute([2,0,1])[None], new_st_grid_coord[None],)
                # mode='nearest')
            rays_color = new_color[0].permute([2,1,0])
            st = new_st

        st = st.reshape([-1,2])
        rays_color = rays_color.reshape([-1,3])
        uv = self.all_uv[idx].expand_as(st)

        if self.batch_size > 0:
            n = st.shape[0]
            perm = torch.randperm(n)
            select_mask = torch.linspace(0,n-1,n)[perm].long()[:self.batch_size]
        else:
            select_mask = torch.ones_like(uv)[:,0].bool()

        rays_ori, rays_dir = uvst2rays(uv[select_mask], st[select_mask], self.plane_dist)

        return {'rays_ori': rays_ori, 
                'rays_dir': rays_dir, 
                'rays_color': rays_color[select_mask], 
                'near': self.near, 
                'far': self.far}

