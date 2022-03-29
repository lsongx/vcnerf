import sys
sys.path += ['../', '../vcnerf']
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
# cfg.data.val.num_render_img = 27*4
dataset = build_dataset(cfg.data.val)
cfg.data.train.batch_size=-1
cfg.data.train.repeat=1
cfg.data.train.pre_shuffle=False
# dataset = build_dataset(cfg.data.train)
render_loader = dataset.render_loader
dataset = render_loader
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
    coarse_depth = result['coarse']['depth_map'].cpu().numpy().reshape([h,w])
    fine_depth = result['fine']['depth_map'].cpu().numpy().reshape([h,w])
    plt.imsave(f'{base}/render-save/pred-coarse-{i:03d}.png', coarse_im)
    plt.imsave(f'{base}/render-save/pred-fine-{i:03d}.png', fine_im)
    plt.imsave(f'{base}/render-save/depth-coarse-{i:03d}.png', coarse_depth)
    plt.imsave(f'{base}/render-save/depth-fine-{i:03d}.png', fine_depth)
    all_coarse_im.append(coarse_im)
    all_fine_im.append(fine_im)
    print(f'[{i:03d}]/[{num_images:03d}] image finished')

imageio.mimsave(f'{base}/render-save/pred-coarse.gif', all_coarse_im)
imageio.mimsave(f'{base}/render-save/pred-fine.gif', all_fine_im)
os.system(f'ffmpeg -y -framerate 8 -i {base}/render-save/pred-fine-%03d.png -b 20M {base}/render-save/pred-fine.mp4')

import ipdb; ipdb.set_trace()


