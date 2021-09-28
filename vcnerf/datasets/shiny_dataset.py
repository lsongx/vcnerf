import os
import glob
import numpy as np
from skimage import io
import torch

from vcnerf.utils import get_root_logger
from .utils.nex_llff_loader import OrbiterDataset
from .builder import DATASETS


@DATASETS.register_module()
class ShinyDataset(object):
    def __init__(self, 
                 base_dir, 
                 downsample,
                 llff_width,
                 batch_size,
                 split,
                 testskip=8,):
        super().__init__()
        self.logger = get_root_logger()
        self.base_dir = os.path.expanduser(base_dir)

        img0 = [os.path.join(self.base_dir, 'images', f) 
                for f in sorted(os.listdir(os.path.join(self.base_dir, 'images')))
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = io.imread(img0).shape
        scale = float(llff_width / sh[1])

        self.orbiter_dataset = OrbiterDataset(self.base_dir, 
                                              ref_img='', 
                                              scale=scale, 
                                              dmin=-1, 
                                              dmax=-1, 
                                              invz=False, 
                                              render_style='shiny',
                                              offset=0,)

    def 

