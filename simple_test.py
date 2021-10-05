import os
import torch
import matplotlib.pyplot as plt
from mmcv import Config
import argparse

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="./data/out/")
args = parser.parse_args()

base = os.path.join(args.f)
for f in os.listdir(base):
    if '.py' in f:
        print(f'{f} loaded.')
        break
cfg_file = f'{base}/{f}'
state_dict = f'{base}/latest.pth'

cfg = Config.fromfile(cfg_file)
val_dataset = build_dataset(cfg.data.val)
dataset = build_dataset(cfg.data.train)
dataset.dataset.batch_size = -1
model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
cfg.evaluation.render_params.max_rays_num = 512
h = val_dataset.h
w = val_dataset.w

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
    coarse_im = result['coarse']['color_map'].cpu().numpy().reshape([h,w,3])
    fine_im = result['fine']['color_map'].cpu().numpy().reshape([h,w,3])
    plt.imsave(f'./data/out/tmp{i}-coarse.png', coarse_im)
    plt.imsave(f'./data/out/tmp{i}-fine.png', fine_im)

    import ipdb; ipdb.set_trace()



