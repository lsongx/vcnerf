from collections import OrderedDict
from typing import final
from vcnerf.datasets.llff_dataset import LLFFDataset
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.runner import auto_fp16, force_fp32
from vcnerf.core import im2mse, mse2psnr, raw2outputs, sample_pdf
from ..builder import RENDERER, build_embedder, build_field


@RENDERER.register_module()
class NeRF(nn.Module):
    def __init__(self, 
                 xyz_embedder, 
                 coarse_field, 
                 render_params,
                 dir_embedder=None, 
                 fine_field=None,):
        super().__init__()
        self.xyz_embedder = build_embedder(xyz_embedder)
        self.coarse_field = build_field(coarse_field)
        self.dir_embedder = build_embedder(dir_embedder)
        self.fine_field = build_field(fine_field)
        self.render_params = render_params
        self.fp16_enabled = False
        self.iter = 0

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if 'loss' not in loss_name:
                continue
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items())
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, rays, render_params=None):
        """        
        Args:
            rays (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: 
        """
        if render_params is None:
            render_params = self.render_params
        outputs = self.forward_render(**rays, **render_params)

        im_loss = im2mse(outputs['coarse']['color_map'], rays['rays_color'])
        outputs['coarse_loss'] = im_loss

        if outputs['fine'] is not None:
            im_loss_fine = im2mse(outputs['fine']['color_map'], rays['rays_color'])
            outputs['fine_loss'] = im_loss_fine

        return outputs

    def _parse_outputs(self, outputs):
        loss, log_vars = self._parse_losses(outputs)
        if outputs['fine'] is not None:
            log_vars['coarse_psnr'] = mse2psnr(outputs['coarse_loss']).item()
            log_vars['psnr'] = mse2psnr(outputs['fine_loss']).item()
        else:
            log_vars['psnr'] = mse2psnr(outputs['coarse_loss']).item()
        outputs.update(dict(loss=loss, log_vars=log_vars))
        outputs['num_samples'] = 1
        return outputs

    @auto_fp16()
    def forward_render(self,
                       rays_ori, rays_dir, rays_color, # loader output
                       n_samples, n_importance, perturb, alpha_noise_std, inv_depth, # render param
                       use_dirs, max_rays_num, near=0.0, far=1.0, white_bkgd=False):
        self.n_importance = n_importance
        self.perturb = perturb
        dtype = rays_dir.dtype
        device = rays_ori.device
        base_shape = list(rays_ori.shape[:-1])
        if isinstance(near, torch.Tensor) and len(near.shape)>0:
            near = near[...,0].item()
            far = far[...,0].item()

        if not inv_depth:
            z_vals = torch.linspace(
                near, far, n_samples, 
                dtype=dtype, device=device).expand([*base_shape, n_samples])
        else:
            # z_vals = torch.linspace(0, 1, n_samples, dtype=dtype, device=device)
            # z_vals = z_vals.pow(0.5).expand([*base_shape, n_samples])
            # z_vals = near*z_vals+far*(1-z_vals)
            z_vals = 1/torch.linspace(
                1, near/far, n_samples, 
                dtype=dtype, device=device).expand([*base_shape, n_samples])*near

        # Perturbs points coordinates
        if perturb:
            # Gets intervals
            mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mid_points, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mid_points], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, dtype=dtype, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        # TODO: double check
        if use_dirs:
            viewdirs = F.normalize(rays_dir, p=2, dim=-1)
            # viewdirs = rays_dir
        else:
            viewdirs = None
        
        # Evaluates the model at the points
        raw2outputs_params = {'alpha_noise_std': alpha_noise_std, 
                              'white_bkgd': white_bkgd,}
        coarse_outputs = self.forward_batchified(rays_ori, 
                                                 rays_dir, 
                                                 z_vals, 
                                                 viewdirs, 
                                                 self.coarse_field,
                                                 raw2outputs_params,
                                                 max_rays_num)

        if n_importance>0 and self.fine_field is not None:
            fine_outputs = self.forward_batchified(rays_ori, 
                                                   rays_dir, 
                                                   z_vals, 
                                                   viewdirs, 
                                                   self.fine_field,
                                                   raw2outputs_params,
                                                   max_rays_num,
                                                   coarse_outputs['weights'][...,1:-1])
        else:
            fine_outputs = None
        return {'fine': fine_outputs, 'coarse': coarse_outputs}

    def sample_new_z_vals(self, z_vals, weights, start=None, end=None):
        z_vals = z_vals[start:end]
        if weights is None:
            return z_vals
        weights = weights[start:end]
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights, 
                               self.n_importance, not self.perturb)
        z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        z_vals_fine = z_vals_fine.detach()
        return z_vals_fine

    @auto_fp16()
    def forward_batchified(self, 
                           rays_ori, rays_dir, z_vals, 
                           viewdirs, field, 
                           raw2outputs_params, max_rays_num, weights=None):
        num_rays = rays_ori.shape[0]
        if num_rays <= max_rays_num or self.training:
            z_vals = self.sample_new_z_vals(z_vals, weights)
            points = rays_ori[..., None, :] + \
                     rays_dir[..., None, :] * \
                     z_vals[..., :, None]
            alphas, colors = self.forward_points(points, viewdirs, field)
            return raw2outputs(alphas, colors, z_vals, rays_dir, **raw2outputs_params)
        else:
            outputs = []
            start = 0
            while start < num_rays:
                end = min(start+max_rays_num, num_rays)
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                local_z_vals = self.sample_new_z_vals(z_vals, weights, start, end)
                points = rays_ori[start:end, None, :] + \
                         rays_dir[start:end, None, :] * \
                         local_z_vals[..., :, None]
                alphas, colors = self.forward_points(points, 
                                                     viewdirs[start: end, ...], 
                                                     field,)
                local_raw2outputs_params = {
                    'z_vals': local_z_vals,
                    'rays_dir': rays_dir[start:end, ...],
                    **raw2outputs_params}
                output = raw2outputs(alphas, colors, **local_raw2outputs_params)
                outputs.append(output)
                start += max_rays_num
            
            final_output = {}
            for k in output.keys():
                if outputs[0][k] is None:
                    final_output[k] = None
                else:
                    final_output[k] = torch.cat([i[k] for i in outputs], 0)
            return final_output

    @auto_fp16(apply_to=('points',))
    def forward_points(self, points, viewdirs, field):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        if self.dir_embedder is None:
            dir_embeds = None
        else:
            assert self.dir_embedder is not None
            viewdirs = viewdirs[..., None, :].expand_as(points)
            viewdirs = viewdirs.reshape((-1, 3))    
            dir_embeds = self.dir_embedder(viewdirs)

        points = points.reshape((-1, 3))
        xyz_embeds = self.xyz_embedder(points)
        alphas, colors = field(xyz_embeds, dir_embeds)
        alphas = alphas.reshape(shape + (1,))
        colors = colors.reshape(shape + (3,))

        return alphas, colors

    def train_step(self, data, optimizer, **kwargs):
        for k, v in data.items():
            if v.shape[0] == 1:
                data[k] = v[0] # batch size = 1
        outputs = self(data, **kwargs)
        outputs = self._parse_outputs(outputs)
        return outputs

    def val_step(self, data, optimizer, **kwargs):
        collect_keys = kwargs.pop('collect_keys', None)
        outputs = self.train_step(data, optimizer, **kwargs)
        if collect_keys is None:
            return outputs
        new_out = {}
        for k in outputs.keys():
            if not isinstance(outputs[k], dict):
                new_out[k] = outputs[k]
                continue
            new_out[k] = {}
            for sub_k in outputs[k].keys():
                if sub_k in collect_keys:
                    new_out[k][sub_k] = outputs[k][sub_k]
        del outputs
        return new_out


