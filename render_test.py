import os
import torch
import matplotlib.pyplot as plt
from mmcv import Config
import imageio

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

for f in os.listdir('./data/out'):
    if 'py' in f:
        break
cfg_file = f'./data/out/{f}'
state_dict = './data/out/latest.pth'

cfg = Config.fromfile(cfg_file)
dataset = build_dataset(cfg.data.val)
dataset.poses = dataset.render_poses
model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
cfg.evaluation.render_params.max_nb_rays = 512
h = dataset.h
w = dataset.w

all_coarse_im = []
all_fine_im = []
num_images = len(dataset)
for i in range(num_images):
    data = {}
    for k, v in dataset[i].items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()
        else:
            data[k] = v
    
    with torch.no_grad():
        result = model.forward_render(**data, **cfg.evaluation.render_params)
    coarse_im = result['coarse']['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    fine_im = result['fine']['color_map'].clamp(0,1).cpu().numpy().reshape([h,w,3])
    plt.imsave(f'./data/out/pred-coarse{i}.png', coarse_im)
    plt.imsave(f'./data/out/pred-fine{i}.png', fine_im)
    all_coarse_im.append(coarse_im)
    all_fine_im.append(fine_im)
    print(f'[{i:03d}]/[{num_images:03d}] image finished')

imageio.mimsave('./data/out/pred-coarse.gif', all_coarse_im)
imageio.mimsave('./data/out/pred-fine.gif', all_fine_im)

import ipdb; ipdb.set_trace()


