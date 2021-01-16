import os
import torch
from mmcv import Config

from vcnerf.models import build_renderer
from vcnerf.datasets import build_dataset

cfg_file = './configs/nerf_synthetic_lego.py'
state_dict = './data/out/latest.pth'

cfg = Config.fromfile(cfg_file)
dataset = build_dataset(cfg.data.val)
model = build_renderer(cfg.model).cuda()
model.load_state_dict(torch.load(state_dict)['state_dict'])
cfg.evaluation.render_params.max_nb_rays = 512

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
    import ipdb; ipdb.set_trace()



