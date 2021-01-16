import os
import os.path as osp

import numpy as np
import cv2
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import mmcv
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate


class EvalHook(Hook):

    def __init__(self,
                 dataloader,
                 render_params,
                 interval=1,
                 out_dir=None,
                 logger=None):
        self.dataloader = dataloader
        self.render_params = render_params
        self.interval = interval
        self.out_dir = out_dir
        self.logger = logger
        self.best_psnr = 0
        self.im_shape = (
            dataloader.dataset.h, dataloader.dataset.w, 3)

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        psnr = self.evaluate(runner)
        is_best = False
        if psnr > self.best_psnr:
            is_best = True
            old_filename = f'checkpoint_{self.best_psnr:.2f}.pth'
            if os.path.isfile(osp.join(self.out_dir, old_filename)):
                os.remove(osp.join(self.out_dir, old_filename))
            self.best_psnr = psnr
            self.bestname = f'checkpoint_{self.best_psnr:.2f}.pth'
            if self.logger is not None:
                self.logger.info(f'Saving best {self.bestname}.')
        torch.save(runner.model.state_dict(), 
                    osp.join(self.out_dir, self.bestname))

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
                 interval=1,
                 out_dir=None,
                 logger=None):
        self.dataloader = dataloader
        self.render_params = render_params
        self.interval = interval
        self.out_dir = out_dir
        self.logger = logger
        self.best_psnr = 0
        self.im_shape = (
            int(dataloader.dataset.h), int(dataloader.dataset.w), 3)

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        psnr = self.evaluate(runner)
        if runner.rank == 0:
            is_best = False
            if psnr > self.best_psnr:
                is_best = True
                old_filename = f'checkpoint_{self.best_psnr:.2f}.pth'
                if os.path.isfile(osp.join(self.out_dir, old_filename)):
                    os.remove(osp.join(self.out_dir, old_filename))
                self.best_psnr = psnr
                self.bestname = f'checkpoint_{self.best_psnr:.2f}.pth'
                if self.logger is not None:
                    self.logger.info(f'Saving best {self.bestname}.')
            torch.save(runner.model.state_dict(), 
                       osp.join(self.out_dir, self.bestname))
        dist.barrier()

    def evaluate(self, runner):
        runner.model.eval()
        loss = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        psnr = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')
        size = torch.zeros(1, dtype=torch.float, device=f'cuda:{runner.rank}')

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
                self.out_dir, 
                f'iter{runner.iter+1}-id{runner.rank+i}-coarse.png'), im[:,:,::-1])
            if outputs['fine'] is not None:
                im = outputs['fine']['color_map'].reshape(self.im_shape)
                im = 255 * im.detach().cpu().numpy()
                # TODO: convert to video
                cv2.imwrite(osp.join(
                    self.out_dir, 
                    f'iter{runner.iter+1}-id{runner.rank+i}-fine.png'), im[:,:,::-1])

            loss += outputs['log_vars']['loss']
            psnr += outputs['log_vars']['psnr']
            size += 1

        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr, op=dist.ReduceOp.SUM)
        dist.all_reduce(size, op=dist.ReduceOp.SUM)
        loss = loss.item()/size.item()
        psnr = psnr.item()/size.item()
        runner.log_buffer.output['loss'] = loss
        runner.log_buffer.output['PSNR'] = psnr
        runner.log_buffer.ready = True
        return psnr
