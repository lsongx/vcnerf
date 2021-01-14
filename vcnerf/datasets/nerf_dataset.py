import numpy as np
import torch
from torch.utils.data import Dataset

from mmcv.utils import build_from_cfg
from .utils import convert_rays_to_ndc_rays
from .builder import DATASETS, LOADERS


@DATASETS.register_module()
class NeRFDataset(Dataset):
    def __init__(self, loader, split, holdout=8):
        super().__init__()
        assert split in ('train', 'val', 'test'), split
        self.loader = build_from_cfg(loader, LOADERS)
        self.split = split
        self.holdout = holdout

        if self.split == 'train':
            rays_ori = [rays_o for i, rays_o in enumerate(self.loader.rays_ori) if i % holdout != 0]
            rays_dir = [rays_d for i, rays_d in enumerate(self.loader.rays_dir) if i % holdout != 0]
            rays_color = [rays_c for i, rays_c in enumerate(self.loader.rays_color) if i % holdout != 0]
            poses = [pose for i, pose in enumerate(self.loader.poses) if i % holdout != 0]
        else:
            rays_ori = [rays_o for i, rays_o in enumerate(self.loader.rays_ori) if i % holdout == 0]
            rays_dir = [rays_d for i, rays_d in enumerate(self.loader.rays_dir) if i % holdout == 0]
            rays_color = [rays_c for i, rays_c in enumerate(self.loader.rays_color) if i % holdout == 0]
            poses = [pose for i, pose in enumerate(self.loader.poses) if i % holdout == 0]
        
        self.rays_ori = np.stack(rays_ori, axis=0).astype('float32')
        self.rays_dir = np.stack(rays_dir, axis=0).astype('float32')
        self.rays_color = np.stack(rays_color, axis=0).astype('float32')
        self.poses = np.stack(poses, axis=0).astype('float32')

        if self.split == 'train':
            self.rays_ori = np.reshape(self.rays_ori, [-1, 3])
            self.rays_dir = np.reshape(self.rays_dir, [-1, 3])
            self.rays_color = np.reshape(self.rays_color, [-1, 3])

        if self.loader.ndc:
            self.ndc_rays_ori, self.ndc_rays_dir = convert_rays_to_ndc_rays(
                h=self.loader.h, 
                w=self.loader.w, 
                focal=self.loader.focal, 
                near=1, 
                rays_o=self.rays_ori,
                rays_d=self.rays_dir
            )

    def __len__(self):
        return self.rays_ori.shape[0]
    
    def __getitem__(self, idx):
        rays_ori = self.rays_ori[idx]  # [3] or [B, 3]
        rays_dir = self.rays_dir[idx]  # [3] or [B, 3]
        rays_color = self.rays_color[idx]  # [3] or [B, 3]

        if self.split != 'train':
            rays_ori = rays_ori.reshape([-1,3])
            rays_dir = rays_dir.reshape([-1,3])
            rays_color = rays_color.reshape([-1,3])

        if self.loader.ndc:
            ndc_rays_ori = self.ndc_rays_ori[idx]
            ndc_rays_dir = self.ndc_rays_dir[idx]
            if self.split != 'train':
                ndc_rays_ori = ndc_rays_ori.reshape([-1,3])
                ndc_rays_dir = ndc_rays_dir.reshape([-1,3])
            return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                    'rays_color': rays_color, 'ndc_rays_ori': ndc_rays_ori, 
                    'ndc_rays_dir': ndc_rays_dir,}

        return {'rays_ori': rays_ori, 'rays_dir': rays_dir, 
                'rays_color': rays_color, 'ndc_rays_ori': 0, 
                'ndc_rays_dir': 0,}


# if __name__ == '__main__':
#     colmap_dir = '/mnt/datasets/NERF/nerf_llff_data/fern/'
#     im_dir = '/mnt/datasets/NERF/nerf_llff_data/fern/' + 'imaGes'.lower()
    
#     nerf_loader = NeRFLoader(colmap_dir, im_dir, factor=8)
#     train_dataset = NeRFDataset(nerf_loader, 'train', holdout=8)
#     train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=False, collate_fn=train_dataset.collater)

#     rays_ori, rays_dir, rays_color, ndc_rays_ori, ndc_rays_dir = next(iter(train_dataloader))
#     import pdb; pdb.set_trace()
#     print('rays_ori:', rays_ori.shape)
#     print('rays_dir:', rays_dir.shape)
#     print('rays_color:', rays_color.shape)
#     print('ndc_rays_ori:', ndc_rays_ori.shape)
#     print('ndc_rays_dir:', ndc_rays_dir.shape)
