import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F
from typing_extensions import override

from .base_trainer import BaseTrainer
from utils import (get_seg_metrics, sampler_set_epoch, get_colormap)
from .seg_trainer import SegTrainer
from .trainer_registry import register_trainer


@register_trainer
class BracsTrainer(SegTrainer):
    def __init__(self, config):
        super().__init__(config)
        if config.is_testing:
            self.colormap = torch.tensor(get_colormap(config)).to(self.device)
        else:
            self.metrics = [get_seg_metrics(config, metric_name).to(self.device) for metric_name in config.metrics]

    @override
    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch)

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (images, masks) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            # Forward path
            if config.use_aux:
                with amp.autocast(enabled=config.amp_training):
                    preds, preds_aux = self.model(images, is_training=True)

                    if config.class_weights is not None: #Pass optional weights from config
                        loss = self.loss_fn(preds, masks, config.class_weights)
                    else:
                        loss = self.loss_fn(preds, masks)
                masks_auxs = masks.unsqueeze(1).float()
                if config.aux_coef is None:
                    config.aux_coef = torch.ones(len(preds_aux))
                elif len(preds_aux) != len(config.aux_coef):
                    raise ValueError('Auxiliary loss coefficient length does not match.')

                for i in range(len(preds_aux)):
                    aux_size = preds_aux[i].size()[2:]
                    masks_aux = F.interpolate(masks_auxs, aux_size, mode='nearest')
                    masks_aux = masks_aux.squeeze(1).to(self.device, dtype=torch.long)

                    with amp.autocast(enabled=config.amp_training):
                        loss += config.aux_coef[i] * self.loss_fn(preds_aux[i], masks_aux)

            else:   # Vanilla forward path
                with amp.autocast(enabled=config.amp_training):
                    preds = self.model(images)
                    #CHNG: ADDED DYNAMIC EXAMPLE SAVE PATH
                    loss = self.loss_fn(preds, masks, save_path = config.save_example_path, class_weights = config.class_weights)

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)

            # Backward path
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.ema_model.update(self.model, self.train_itrs)

            if self.main_rank:
                pbar.set_description(('%s'*2) %
                                (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" "*4}|',
                                f'Loss:{loss.detach():4.4g}{" "*4}|',)
                                )

        return

    @torch.no_grad()
    def validate(self, config, loader, val_best=False):
        pbar = tqdm(loader) if self.main_rank else loader
        for (images, masks) in pbar:
            images = images.to(self.device, dtype=torch.float32)

            # In order to validate different size of images, we can resize the image to be compatible with the model's stride,
            # and then resize the predicted mask back to the image size to calculate the metric. 
            # If you don't want this, simply set `val_img_stride=1` within the config file.
            _, _, H, W = images.shape
            stride = config.val_img_stride
            if H % stride != 0 or W % stride != 0:
                new_size = (H // stride * stride, W // stride * stride)
                images = F.interpolate(images, new_size, mode='bilinear')

            masks = masks.to(self.device, dtype=torch.long)

            #PERMUTE MASKS TO [B C H W]
            masks = masks.permute(0, 3, 1, 2)


            preds = self.ema_model.ema(images)
            if H % stride != 0 or W % stride != 0:
                preds = F.interpolate(preds, masks.size()[1:], mode='bilinear', align_corners=True)


            #Torch metrics expects INTEGER mask classes
            # Convert predictions to class indices
            preds = torch.argmax(preds, dim=1)  # shape: [B, H, W]

            # Convert one-hot target to class indices if needed
            if masks.ndim == 4 and masks.shape[1] > 1:
                masks = torch.argmax(masks, dim=1)  # shape: [B, H, W]

            for metric in self.metrics:
                metric.update(preds.detach(), masks)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        scores = [metric.compute() for metric in self.metrics]
        score = scores[0].mean()

        if self.main_rank:
            for i in range(len(config.metrics)):
                if val_best:
                    self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.' +
                                     f'\n\nBest m{config.metrics[i]} is: {scores[i].mean():.4f}\n')
                else:
                    infos = f' Epoch{self.cur_epoch} m{config.metrics[i]}: {scores[i].mean():.4f} \t| ' + \
                                     f'best m{config.metrics[0]} so far: {self.best_score:.4f}\n'
                    if len(config.metrics) > 1 and i != len(config.metrics) - 1:
                        infos = infos[:-1]
                    self.logger.info(infos)

                if config.use_tb and self.cur_epoch < config.total_epoch:
                    self.writer.add_scalar(f'val/m{config.metrics[i]}', scores[i].mean().item(), self.cur_epoch+1)
                    if config.metrics[i] == 'iou':
                        for j in range(config.num_class):
                            self.writer.add_scalar(f'val/IoU_cls{j:02f}', scores[i][j].item(), self.cur_epoch+1)
        for metric in self.metrics:
            metric.reset()
        return score

    @override
    def load_ckpt(self, config):
        if config.load_ckpt and os.path.isfile(config.load_ckpt_path):
            checkpoint = torch.load(config.load_ckpt_path, map_location=torch.device(self.device))

            '''
            Loading checkpoints HERE. 
            '''
            # self.model.load_state_dict(checkpoint['state_dict'])

            ###################
            state_dict = checkpoint["state_dict"]

            # Remove any segmentation head keys that no longer match
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("seg_head.") and not k.startswith("seg_head_bracs.")
            }

            missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)

            print("[INFO] Loaded pretrained weights (excluding final segmentation head).")
            print("[INFO] Missing keys:", missing)
            print("[INFO] Unexpected keys:", unexpected)
            ####################

            if self.main_rank:
                self.logger.info(f"Load model state dict from {config.load_ckpt_path}")

            if not config.is_testing and config.resume_training:
                self.cur_epoch = checkpoint['cur_epoch'] + 1
                self.best_score = checkpoint['best_score']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.train_itrs = self.cur_epoch * config.iters_per_epoch
                if self.main_rank:
                    self.logger.info(f"Resume training from {config.load_ckpt_path}")

            del checkpoint
        else:
            if config.is_testing:
                raise ValueError(f'Could not find any pretrained checkpoint at path: {config.load_ckpt_path}.')
            else:
                if self.main_rank:
                    self.logger.info('[!] Train from scratch')