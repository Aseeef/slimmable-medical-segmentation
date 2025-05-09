import statistics
import time

import torch
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from utils import (get_seg_metrics, sampler_set_epoch, get_colormap)
from .trainer_registry import register_trainer


@register_trainer
class SegTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if config.is_testing:
            self.colormap = torch.tensor(get_colormap(config)).to(self.device)
        else:
            self.metrics = [get_seg_metrics(config, metric_name).to(self.device) for metric_name in config.metrics]

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
                    loss = self.loss_fn(preds, masks)

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
        scores = self.val_compute_metrics(config, pbar)

        if self.main_rank:
            for i in range(len(config.metrics)):
                if val_best:
                    self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.' +
                                     f'\n\nBest m{config.metrics[i]} is: {scores[i].mean():.4f}\n')
                else:
                    infos = f' Epoch {self.cur_epoch} m{config.metrics[i]}: {scores[i].mean():.4f} \t| ' + \
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

        return scores[0].mean()

    def val_compute_metrics(self, config, pbar):
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

            preds = self.ema_model.ema(images)
            if H % stride != 0 or W % stride != 0:
                preds = F.interpolate(preds, masks.size()[1:], mode='bilinear', align_corners=True)

            for metric in self.metrics:
                metric.update(preds.detach(), masks)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        scores = [metric.compute() for metric in self.metrics]

        return scores

    @torch.no_grad()
    def predict(self, config):
        if config.DDP:
            raise ValueError('Predict mode currently does not support DDP.')

        self.logger.info('\nStart predicting...\n')

        self.model.eval() # Put model in evalation mode

        pred_times = []

        for (images, images_aug, img_names) in tqdm(self.test_loader):
            images_aug = images_aug.to(self.device, dtype=torch.float32)

            start = time.time()
            preds = self.model(images_aug)
            end = time.time()
            pred_times.append(end - start)

            preds = self.colormap[preds.max(dim=1)[1]].cpu().numpy()

            images = images.cpu().numpy()

            # Saving results
            for i in range(preds.shape[0]):
                save_path = os.path.join(config.save_dir, img_names[i])
                save_suffix = img_names[i].split('.')[-1]

                pred = Image.fromarray(preds[i].astype(np.uint8))

                if config.save_mask:
                    pred.save(save_path)

                if config.blend_prediction:
                    save_blend_path = save_path.replace(f'.{save_suffix}', f'_blend.{save_suffix}')

                    image = Image.fromarray(images[i].astype(np.uint8))
                    image = Image.blend(image, pred, config.blend_alpha)
                    image.save(save_blend_path)
            print(statistics.mean(pred_times))