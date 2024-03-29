import os
import torch
import matplotlib.pyplot as plt
from mmcv import Config
import imageio
import argparse

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="./data/out/")
parser.add_argument("-c", default=None)
args = parser.parse_args()

base = os.path.join(args.f)
for f in os.listdir(base):
    if '.py' in f:
        print(f'{f} loaded.')
        break
cfg_file = f'{base}/{f}'
if args.c is None:
    args.c = f'{base}/latest.pth'

if os.path.isdir(f'{base}/render-save'):
    os.system(f'rm -rf {base}/render-save/*')
else:
    os.mkdir(f'{base}/render-save')

cfg = Config.fromfile(cfg_file)
# cfg.data.val.llff_data_param = {'percentile': 38, 'N_views': 27*3, 'zrate': 0.8}
cfg.data.val.llff_data_param = {'percentile': 38, 'N_views': 27, 'N_rots':1, 'zrate': 0.8}
dataset = build_dataset(cfg.data.val)
dataset.poses = dataset.render_poses
model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(args.c)['state_dict'])
cfg.evaluation.render_params.max_rays_num = 512
h = dataset.h
w = dataset.w

all_coarse_im = []
all_fine_im = []
model.eval()
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
    plt.imsave(f'{base}/pred-coarse-{i:03d}.png', coarse_im)
    plt.imsave(f'{base}/pred-fine-{i:03d}.png', fine_im)
    all_coarse_im.append(coarse_im)
    all_fine_im.append(fine_im)
    print(f'[{i:03d}]/[{num_images:03d}] image finished')

imageio.mimsave(f'{base}/pred-coarse.gif', all_coarse_im)
imageio.mimsave(f'{base}/pred-fine.gif', all_fine_im)
os.system(f'ffmpeg -y -framerate 8 -i {base}/pred-fine-%03d.png -b 20M {base}/pred-fine.avi')

import ipdb; ipdb.set_trace()


