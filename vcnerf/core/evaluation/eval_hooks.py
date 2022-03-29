from genericpath import isfile
import os
import os.path as osp
from unittest.mock import patch
from matplotlib import pyplot as plt

import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import mmcv
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import lpips


class EvalHook(Hook):

    def __init__(self,
                 dataloader,
                 render_params,
                 epoch_interval=1,
                 iter_interval=5e3,
                 out_dir=None,
                 logger=None):
        self.dataloader = dataloader
        self.render_params = render_params
        self.epoch_interval = epoch_interval
        self.iter_interval = iter_interval
        self.out_dir = out_dir
        self.logger = logger
        self.best_psnr = 0
        self.im_shape = (
            dataloader.dataset.h, dataloader.dataset.w, 3)

    # def after_train_epoch(self, runner):
    #     if not self.every_n_epochs(runner, self.epoch_interval):
    #         return
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.iter_interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        psnr = self.evaluate(runner)
        if psnr > self.best_psnr:
            old_filename = f'checkpoint_{self.best_psnr:.2f}.pth'
            if os.path.isfile(osp.join(self.out_dir, old_filename)):
                os.remove(osp.join(self.out_dir, old_filename))
            self.best_psnr = psnr
            self.bestname = f'checkpoint_{self.best_psnr:.2f}.pth'
            if self.logger is not None:
                self.logger.info(f'Saving best {self.bestname}.')
            torch.save(runner.model.state_dict(), 
                        osp.join(self.out_dir, self.bestname))
        else:
            self.logger.info(f'Current best {self.bestname}.')

    def evaluate(self, runner):
        runner.model.eval()
        loss = 0
        psnr = 0
        size = 0

        for i, rays in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = runner.model.val_step(rays, 
                                                runner.optimizer,
                                                render_params=self.render_params)

            # save images
            im = outputs['coarse']['color_map'].reshape(self.im_shape)
            im = 255 * im.detach().cpu().numpy()
            # TODO: convert to video
            cv2.imwrite(osp.join(
                self.out_dir, f'iter{runner.iter+1}-id{i}-coarse.png'), im[:,:,::-1])
            if outputs['fine'] is not None:
                im = outputs['fine']['color_map'].reshape(self.im_shape)
                im = 255 * im.detach().cpu().numpy()
                # TODO: convert to video
                cv2.imwrite(osp.join(
                    self.out_dir, f'iter{runner.iter+1}-id{i}-fine.png'), im[:,:,::-1])

            loss += outputs['log_vars']['loss']
            psnr += outputs['log_vars']['psnr']
            size += 1

        loss = loss / size
        psnr = psnr / size
        runner.log_buffer.output['loss'] = loss
        runner.log_buffer.output['PSNR'] = psnr
        runner.log_buffer.ready = True
        return psnr


class DistEvalHook(Hook):

    def __init__(self,
                 dataloader,
                 render_params,
                 epoch_interval=1,
                 iter_interval=5e3,
                 out_dir=None,
                 logger=None):
        self.dataloader = dataloader
        self.render_params = render_params
        self.epoch_interval = epoch_interval
        self.iter_interval = iter_interval
        self.out_dir = out_dir
        self.logger = logger
        self.best_psnr = 0
        self.bestname = f'checkpoint_{self.best_psnr:.2f}.pth'
        self.im_shape = (
            int(dataloader.dataset.h), int(dataloader.dataset.w), 3)
        self.lpips_model = lpips.LPIPS(net='vgg')

    # def after_train_epoch(self, runner):
    #     if not self.every_n_epochs(runner, self.epoch_interval):
    #         return
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.iter_interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        psnr = self.evaluate(runner)
        if runner.rank == 0:
            if psnr > self.best_psnr:
                old_filename = f'checkpoint_{self.best_psnr:.2f}.pth'
                if os.path.isfile(osp.join(self.out_dir, old_filename)):
                    os.remove(osp.join(self.out_dir, old_filename))
                self.best_psnr = psnr
                self.bestname = f'checkpoint_{self.best_psnr:.2f}.pth'
                if self.logger is not None:
                    self.logger.info(f'Saving best {self.bestname}.')
                torch.save(runner.model.state_dict(), 
                           osp.join(self.out_dir, self.bestname))
            else:
                self.logger.info(f'Current best {self.bestname}.')
        dist.barrier()

    def evaluate(self, runner):
        runner.model.eval()
        loss = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        psnr = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        ssim_score = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        lpips_score = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        size = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        step = runner.iter+1 if runner.iter is not None else 0

        for i, rays in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = runner.model.val_step(rays, 
                                                runner.optimizer,
                                                render_params=self.render_params,
                                                collect_keys=['color_map', 'depth_map', 
                                                              'loss', 'psnr'])

            save_dir = osp.join(self.out_dir, f'iter{step}-id{runner.rank+i}')
            # save images
            im_ori = outputs['coarse']['color_map'].reshape(self.im_shape)
            im = 255 * im_ori.detach().cpu().numpy()
            cv2.imwrite(save_dir+'-coarse.png', im[:,:,::-1])
            plt.imsave(save_dir+'-coarse-depth.png', 
                       outputs['coarse']['depth_map'].cpu().numpy().reshape(self.im_shape[:2]))
            if outputs['fine'] is not None:
                im_ori_fine = outputs['fine']['color_map'].reshape(self.im_shape)
                im = 255 * im_ori_fine.detach().cpu().numpy()
                cv2.imwrite(save_dir+'-fine.png', im[:,:,::-1])
                plt.imsave(save_dir+'-fine-depth.png', 
                           outputs['fine']['depth_map'].cpu().numpy().reshape(self.im_shape[:2]))
            gt_path = osp.join(self.out_dir, f'gt-id{runner.rank+i}.png')
            gt_ori = rays['rays_color'].reshape(self.im_shape)
            gt = 255 * gt_ori.detach().cpu().numpy()
            if not os.path.isfile(gt_path):
                cv2.imwrite(gt_path, gt[:,:,::-1])

            gt_lpips = gt_ori.cpu().permute([2,0,1]) * 2.0 - 1.0
            predict_image_lpips = im_ori_fine.cpu().permute([2,0,1]).clamp(0,1) * 2.0 - 1.0
            lpips_score += self.lpips_model.forward(predict_image_lpips, gt_lpips).cpu().detach().item()

            gt_load = tf.image.decode_image(tf.io.read_file(osp.join(self.out_dir, f'gt-id{runner.rank+i}.png')))
            pred_load = tf.image.decode_image(tf.io.read_file(save_dir+'-fine.png'))
            gt_load = tf.expand_dims(gt_load, axis=0)
            pred_load = tf.expand_dims(pred_load, axis=0)
            ssim = tf.image.ssim(gt_load, pred_load, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
            ssim_score += float(ssim[0])

            loss += outputs['log_vars']['loss']
            psnr += outputs['log_vars']['psnr']
            size += 1

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(size, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_score, op=dist.ReduceOp.SUM)
        dist.all_reduce(lpips_score, op=dist.ReduceOp.SUM)
        loss_v = loss.item()/size.item()
        psnr_v = psnr.item()/size.item()
        ssim_v = ssim_score.item()/size.item()
        lpips_v = lpips_score.item()/size.item()
        self.logger.info(f'loss {loss_v:.3f}   '
                         f'psnr {psnr_v:.3f}   '
                         f'ssim {ssim_v:.3f}   '
                         f'lpips {lpips_v:.3f}   ')
        return psnr_v
