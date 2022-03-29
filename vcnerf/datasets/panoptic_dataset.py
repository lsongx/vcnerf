from math import ceil
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import json
import os
import random
import numpy as np
import torch

from vcnerf.utils import get_root_logger
from .builder import DATASETS


def unproj(pt2d, K, distCoef, R=None, t=None):
    # pt2d: [2, N]
    # distCoef: [5]
    # R: w2c; t: w2c; R.T@(p-t)
    N_points = pt2d.shape[1]
    pt2d = torch.tensor(pt2d)
    K_inv = torch.linalg.inv(torch.tensor(K).to(pt2d))
    unproj_pt = K_inv @ torch.cat([pt2d, torch.ones([1,N_points]).to(pt2d)], dim=0)
    k = torch.zeros([12])
    k[:5] = torch.tensor(distCoef)[:5]
    k.to(pt2d)
    x0, y0 = unproj_pt[0,:], unproj_pt[1,:]
    x, y = x0.clone(), y0.clone()
    for _ in range(5):
        r2 = x*x + y*y
        icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2
        deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2
        x = (x0 - deltaX)*icdist
        y = (y0 - deltaY)*icdist
    if R is None:
        unproj_pt = torch.stack([x,y])
        return unproj_pt
    unproj_pt = torch.stack([x,y,torch.ones_like(x)]) # depth=1
    rays_dir = torch.tensor(R).to(pt2d).T @ unproj_pt
    rays_ori = -torch.tensor(R).to(pt2d).T @ torch.tensor(t).to(pt2d).view([3,1])
    return rays_ori.expand_as(rays_dir), rays_dir


@DATASETS.register_module()
class PanopticDataset:
    def __init__(self, 
                 datadir, 
                 batch_size,
                 selected_cam,
                 ratio=1,
                 frame=0,
                 pre_shuffle=True,
                 repeat=int(1e6),
                 num_render_img=27*3,
                 to_cuda=False):
        self.logger = get_root_logger()
        self.batch_size = batch_size
        datadir = os.path.expanduser(datadir)

        all_cams = [i for i in os.listdir(f'{datadir}/hdImgs/') if 'x' not in i]
        self.ratio = ratio
        self.selected_time = [frame]
        img_root_path = f'{datadir}/hdImgs/{int(1/ratio)}x'
        self.ori_w, self.ori_h = Image.open(
            f'{datadir}/hdImgs/{all_cams[0]}/'
            f'{all_cams[0]}_{self.selected_time[0]:08d}.jpg').size
        self.w = int(self.ori_w*self.ratio)
        self.h = int(self.ori_h*self.ratio)
        def resize_all():
            invalid = []
            self.logger.info(f'resize all and checking invalid')
            for cam in all_cams:
                previous = None
                for t in self.selected_time:
                    im = Image.open(f'{datadir}/hdImgs/{cam}/{cam}_{t:08d}.jpg')\
                              .resize((self.w,self.h), Image.BILINEAR)
                    im.save(f'{img_root_path}/{cam}_{t:08d}.jpg')
                    if previous is not None:
                        diff = np.abs((np.array(im)-previous)).sum(2)
                        if (diff>300).sum() > self.w*self.h*0.5:
                            invalid.append(f'{datadir}/hdImgs/{cam}/{cam}_{t:08d}.jpg')
                            continue
                    previous = np.array(im)
            with open(f'{img_root_path}/invalid.json', 'w') as f:
                json.dump(invalid, f)
            self.logger.info(f'resize all finished with invalid {invalid}')
        if not os.path.isdir(f'{img_root_path}'): # resize images
            os.mkdir(f'{img_root_path}')
            resize_all()

        self.selected_cams = [all_cams[i] for i in selected_cam]
        self.all_hd, self.all_img, self.all_time, self.all_cam = [], [], [], []
        for t in self.selected_time:
            self.all_hd += [f'{datadir}/hdImgs/{cam}/{cam}_{t:08d}.jpg'
                            for cam in self.selected_cams]
            self.all_img += [f'{img_root_path}/{cam}_{t:08d}.jpg'
                             for cam in self.selected_cams]
            self.all_cam += self.selected_cams
            self.all_time += [t]*len(self.selected_cams)
        if not all(map(os.path.isfile, self.all_img)):
            resize_all()
        with open(f'{img_root_path}/invalid.json', 'r') as f:
            invalid = json.load(f)
        self.all_img = [i for i in self.all_img if i not in invalid]

        # cameras
        seq_name = datadir.split('/')[-1] if datadir[-1] != '/' else datadir.split('/')[-2]
        with open(f'{datadir}/calibration_{seq_name}.json') as cfile:
            calib = json.load(cfile)
        cameras_raw = {f"{cam['panel']:02d}_{cam['node']:02d}":cam 
                       for cam in calib['cameras']}
        self.cam_params = {}
        for cam in self.selected_cams:
            tmp_cam = cameras_raw[cam]
            self.cam_params[cam] = dict(
                K=torch.tensor(tmp_cam['K']).float(),
                distCoef=torch.tensor(tmp_cam['distCoef']).float(),
                R=torch.tensor(tmp_cam['R']).float(),
                t=torch.tensor(tmp_cam['t']).float(),)
        def get_w2c(R, t):
            tmp = torch.zeros([3,4])
            tmp[:3,:3] = torch.tensor(R)
            tmp[:3,3] = torch.tensor(t).view([-1])
            return tmp
        self.all_w2c = [get_w2c(self.cam_params[cam]['R'], self.cam_params[cam]['t']) 
                        for cam in self.all_cam]
        self.all_K = [self.cam_params[cam]['K'] for cam in self.all_cam]
        self.near = 40
        self.far = 600
        self.coord_scale = 300

        self.length = len(self.all_img)
        self.repeat = repeat
        self.to_cuda = to_cuda
        self.pre_shuffle = pre_shuffle
        if to_cuda:
            self.logger.info(f'start caching...')
            self.all_data, self.all_fg_data = [], []
            perm = torch.arange(self.h*self.w)
            if self.pre_shuffle:
                perm = torch.randperm(int(self.h*self.w))
            for i in range(len(self.all_img)):
                data = self.load_raw(i)
                data_new = {}
                for k,v in data.items():
                    if v.shape[0]>self.h:
                        data_new[k] = v.cuda()[perm]
                    else:
                        data_new[k] = v.cuda()
                self.all_data.append(data_new)
            self.logger.info(f'caching finished')
        self.render_loader = RenderLoader(self.h, self.w, self.ori_h, self.ori_w, 
                                          self.near, self.far, self.coord_scale,
                                          num_render_img, self.cam_params)
        self.logger.info(
            f'total {len(self.all_img)} images '
            f'with {len(self.selected_cams)} cams '
            f'({len(self.all_img)*self.h*self.w:d} rays) loaded')
        self.iter = 0

    def __len__(self):
        return int(self.length*self.repeat)

    def __getitem__(self, idx):
        idx = idx % self.length
        return self.load_idx(idx, self.batch_size) 

    def load_idx(self, idx, length):
        if self.to_cuda:
            if self.pre_shuffle and length>0:
                all_data = {}
                s = int(random.random()*(self.h*self.w-length))
                for k,v in self.all_data[idx].items():
                    if v.shape[0] > self.h: # other inputs like near far
                        all_data[k] = v[s:s+length]
                    else:
                        all_data[k] = v
                return all_data
            else:
                all_data = self.all_data[idx]
        else:
            all_data = self.load_raw(idx)
        return all_data

    def load_raw(self, idx):
        # image = Image.open(self.all_img[idx]).resize([self.w,self.h], Image.BILINEAR)
        image = Image.open(self.all_img[idx])
        cam_param = self.cam_params[self.all_cam[idx]]
        i, j = torch.meshgrid(torch.linspace(0, self.w-1, self.w)*(self.ori_w//self.w), 
                              torch.linspace(0, self.h-1, self.h)*(self.ori_h//self.h),)
        i, j = i.T, j.T
        coords = torch.stack([i,j], dim=-1).view([-1,2]).T # [2,N]
        rays_ori, rays_dir = unproj(coords, **cam_param)
        rays_ori, rays_dir = rays_ori.T, rays_dir.T
        rays_color = torch.tensor(np.array(image)).view([-1,3])/255.
        return {'rays_ori': rays_ori/self.coord_scale, 
                'rays_dir': rays_dir, 
                'rays_color': rays_color, 
                'near': torch.tensor(self.near).float().view([-1])/self.coord_scale, 
                'far': torch.tensor(self.far).float().view([-1])/self.coord_scale,}


class RenderLoader:
    def __init__(self, h, w, ori_h, ori_w,
                 near, far, coord_scale,
                 num_poses, train_cams) -> None:
        self.h = h
        self.w = w
        self.ori_h = ori_h
        self.ori_w = ori_w
        self.near = near
        self.far = far
        self.coord_scale = coord_scale
        self.num_poses = num_poses
        all_position = []
        for cam in train_cams.values():
            all_position.append(-cam['R'].T@cam['t'])
        all_position = torch.stack(all_position)[:,1]
        mean_cam = (all_position-all_position.mean()).abs().argmin()
        self.default_cam = list(train_cams.values())[mean_cam]
        self.default_cam_id = list(train_cams.keys())[mean_cam]
        theta = [2*np.pi/num_poses*k for k in range(num_poses)]
        # up ground axis is the y-axis
        self.rotates = [
            torch.tensor([[np.cos(i), 0, np.sin(i)], 
                          [0, 1, 0],
                          [-np.sin(i), 0, np.cos(i)],]).float()
            for i in theta]
        self.render_w2c = []
        for r in self.rotates:
            tmp = torch.zeros([3,4])
            tmp[:3,:3] = self.default_cam['R'] @ r.T
            tmp[:3,3] = self.default_cam['t'].view([-1])
            self.render_w2c.append(tmp)
        self.K = self.default_cam['K']

    def proj2d(self, X, i=None, w2c=None, K=None):
        # X [3,N]
        X = torch.tensor(X)*self.coord_scale
        if i is not None:
            R = self.default_cam['R'] @ self.rotates[i].T
            t = self.default_cam['t'].view([3,1])
            K = self.default_cam['K']
        else:
            R = torch.tensor(w2c[:3,:3])
            t = torch.tensor(w2c[:3,3]).view([3,1])
            Kd = torch.zeros_like(self.default_cam['distCoef'])
        Kd = self.default_cam['distCoef']
        x = R@X + t
        x[0:2,:] = x[0:2,:]/x[2,:]
        r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
        x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
        x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])
        x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
        x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
        x[0,:] /= self.ori_h/self.h
        x[1,:] /= self.ori_w/self.w
        return x[:2,:].T # [N,2]

    # for compatibility
    def set_frame(self, *args, **kwargs):
        return

    def __len__(self):
        return self.num_poses
    
    def __getitem__(self, idx):
        cam_param = {
            **self.default_cam,
            'R': self.default_cam['R']@self.rotates[idx].T,}
        i, j = torch.meshgrid(torch.linspace(0, self.w-1, self.w)*(self.ori_w//self.w), 
                              torch.linspace(0, self.h-1, self.h)*(self.ori_h//self.h),)
        i, j = i.T, j.T
        
        coords = torch.stack([i,j], dim=-1).view([-1,2]).T # [2,N]
        rays_ori, rays_dir = unproj(coords, **cam_param)
        rays_ori, rays_dir = rays_ori.T, rays_dir.T
        rays_color = torch.zeros_like(rays_ori)
        return {'rays_ori': rays_ori/self.coord_scale, 
                'rays_dir': rays_dir, 
                'rays_color': rays_color, 
                'near': torch.tensor(self.near).float().view([-1])/self.coord_scale, 
                'far': torch.tensor(self.far).float().view([-1])/self.coord_scale,}


    def get(self, w2c, K):
        cam_param = {
            **self.default_cam,
            'R': torch.tensor(w2c[:3,:3]),
            't': torch.tensor(w2c[:3,3]).view([3,1]),
            'K': torch.tensor(K)}
        i, j = torch.meshgrid(torch.linspace(0, self.w-1, self.w)*(self.ori_w//self.w), 
                              torch.linspace(0, self.h-1, self.h)*(self.ori_h//self.h),)
        i, j = i.T, j.T

        coords = torch.stack([i,j], dim=-1).view([-1,2]).T # [2,N]
        rays_ori, rays_dir = unproj(coords, **cam_param)
        rays_ori, rays_dir = rays_ori.T, rays_dir.T
        rays_color = torch.zeros_like(rays_ori)
        return {'rays_ori': rays_ori/self.coord_scale, 
                'rays_dir': rays_dir, 
                'rays_color': rays_color, 
                'near': torch.tensor(self.near).float().view([-1])/self.coord_scale, 
                'far': torch.tensor(self.far).float().view([-1])/self.coord_scale,}


