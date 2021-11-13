import os
import torch
import matplotlib.pyplot as plt
from mmcv import Config
import argparse
import time

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="./data/out/")
parser.add_argument("-m", default="latest.pth")
args = parser.parse_args()

base = os.path.join(args.f)
for f in os.listdir(base):
    if '.py' in f:
        print(f'{f} loaded.')
        break
cfg_file = f'{base}/{f}'
state_dict = f'{base}/{args.m}'

cfg = Config.fromfile(cfg_file)
val_dataset = build_dataset(cfg.data.val)
# dataset = build_dataset(cfg.data.train)
# dataset.dataset.batch_size = -1
model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
print(f'{state_dict} used.')
cfg.evaluation.render_params.max_rays_num = 512
h = val_dataset.h
w = val_dataset.w

test_dataset = val_dataset
near = test_dataset.near
far = test_dataset.far
num_images = len(test_dataset)
model.eval()
for i in range(num_images):
    data = {}
    for k, v in test_dataset[i].items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()
        else:
            data[k] = v
    
    start = time.time()
    with torch.no_grad():
        result = model.forward_render(**data, **cfg.evaluation.render_params)
    run_time = time.time()-start
    coarse_im = result['coarse']['color_map'].cpu().numpy().reshape([h,w,3]).clip(0,1)
    coarse_depth = result['coarse']['depth_map'].cpu().numpy().reshape([h,w,1])
    coarse_acc = result['coarse']['acc_map'].cpu().numpy().reshape([h,w,1])
    fine_im = result['fine']['color_map'].cpu().numpy().reshape([h,w,3]).clip(0,1)
    fine_depth = result['fine']['depth_map'].cpu().numpy().reshape([h,w,1])
    fine_acc = result['fine']['acc_map'].cpu().numpy().reshape([h,w,1])
    # plt.imsave(f'{base}/tmp{i}-coarse.png', coarse_im)
    # plt.imsave(f'{base}/tmp{i}-fine.png', fine_im)
    fig, axes = plt.subplots(2, 2, figsize=(8,8), dpi=256)
    axes[0,0].imshow(coarse_im); axes[0,0].set_title('coarse_im')
    axes[0,1].imshow(fine_im); axes[0,1].set_title('fine_im')
    im0 = axes[1,0].imshow(coarse_depth, vmin=near, vmax=far); axes[1,0].set_title('coarse_depth')
    im1 = axes[1,1].imshow(fine_depth, vmin=near, vmax=far); axes[1,1].set_title('fine_depth')
    # axes[1,0].imshow(coarse_acc); axes[1,0].set_title('coarse_acc')
    # axes[1,1].imshow(fine_acc); axes[1,1].set_title('fine_acc')
    fig.savefig(f'{base}/tmp{i}.png', format='png')
    print(f'runtime {run_time}')
    import ipdb; ipdb.set_trace()





