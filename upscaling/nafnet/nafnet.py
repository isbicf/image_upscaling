import sys

import numpy as np
import torch

# NAFNet modules path added in __init__.py
from basicsr.demo_ssr import parse_options
from basicsr.models import create_model
from basicsr.utils import img2tensor, tensor2img

from upscaling import project_home


class InvalidScale(RuntimeError):
    pass


class Debluring:
    model_path = str(project_home.joinpath('NAFNet', 'experiments', 'pretrained_models', 'NAFNet-REDS-width64.pth'))
    options = {
        'model_type': 'ImageRestorationModel',
        'scale': 1,
        'num_gpu': torch.cuda.device_count(),
        'network_g': {
            'type': 'NAFNetLocal',
            'width': 64,
            'enc_blk_nums': [1, 1, 1, 28],
            'dec_blk_nums': [1, 1, 1, 1]
        },
        'path': {
            'pretrain_network_g': model_path,
        },
        'val': {
            'grids': False,
        },
        'is_train': False,
        'dist': False,
    }

    def __init__(self):
        self.model = create_model(Debluring.options)

    def deblur(self, img):
        img_tensor = img2tensor(img.astype(np.float32) / 255)
        self.model.feed_data(data={'lq': img_tensor.unsqueeze(dim=0)})

        if self.model.opt['val'].get('grids'):
            self.model.grids()
        self.model.test()
        if self.model.opt['val'].get('grids'):
            self.model.grids_inverse()
        visuals = self.model.get_current_visuals()

        return tensor2img([visuals['result']])


class SuperResolution:
    model_path = str(project_home.joinpath('NAFNet', 'experiments', 'pretrained_models', 'NAFSSR-L_4x.pth'))
    options = {
        'model_type': 'ImageRestorationModel',
        'scale': 4,
        'num_gpu': torch.cuda.device_count(),
        'network_g': {
            'type': 'NAFSSR',
            'width': 128,
            'num_blks': 128
        },
        'path': {
            'pretrain_network_g': model_path,
        },
        'val': {
            'grids': False,
        },
        'is_train': False,
        'dist': False,
    }
    max_images = 2
    
    def __init__(self):
        self.model = create_model(SuperResolution.options)

    def upscale(self, imgs):
        if isinstance(imgs, list):
            if len(imgs) > SuperResolution.max_images:
                raise
            img_tensor = img2tensor([img.astype(np.float32) / 255 for img in imgs])
        else:
            img_tensor = img2tensor([imgs.astype(np.float32) / 255, np.zeros(imgs.shape, dtype=np.float32) / 255])
        input_tensors = torch.cat(img_tensor, dim=0)
        self.model.feed_data(data={'lq': input_tensors.unsqueeze(dim=0)})

        if self.model.opt['val'].get('grids'):
            self.model.grids()
        self.model.test()
        if self.model.opt['val'].get('grids'):
            self.model.grids_inverse()
        visuals = self.model.get_current_visuals()
        upscaled_tensor = visuals['result'][:, :3]

        return tensor2img([upscaled_tensor, ])
