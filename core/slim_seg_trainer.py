from enum import Enum

import torch
from torch.cuda import amp
from tqdm import tqdm
from typing_extensions import override

from core import SegTrainer
from core.trainer_registry import register_trainer
from utils import sampler_set_epoch


# TODO: experiment with layer
class SlimmableTrainingType(Enum):
    NONE = "none"
    S_NET = "s-net"
    US_NET = "us-net"

@register_trainer
class SlimmableSegTrainer(SegTrainer):
    def __init__(self, config):
        super().__init__(config)

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
                # TODO: maybe something we can experiment with later
                raise NotImplementedError("Auxiliary loss not implemented yet.")
                ...
                #
                # with amp.autocast(enabled=config.amp_training):
                #     preds, preds_aux = self.model(images, is_training=True)
                #     loss = self.loss_fn(preds, masks)
                #
                # masks_auxs = masks.unsqueeze(1).float()
                # if config.aux_coef is None:
                #     config.aux_coef = torch.ones(len(preds_aux))
                # elif len(preds_aux) != len(config.aux_coef):
                #     raise ValueError('Auxiliary loss coefficient length does not match.')
                #
                # for i in range(len(preds_aux)):
                #     aux_size = preds_aux[i].size()[2:]
                #     masks_aux = F.interpolate(masks_auxs, aux_size, mode='nearest')
                #     masks_aux = masks_aux.squeeze(1).to(self.device, dtype=torch.long)
                #
                #     with amp.autocast(enabled=config.amp_training):
                #         loss += config.aux_coef[i] * self.loss_fn(preds_aux[i], masks_aux)

            else:  # Vanilla forward path
                total_loss = 0.0
                with amp.autocast(enabled=config.amp_training):
                    for w in config.slim_width_mult_list:
                        # Switch the batch normalization parameters of current width on network M.
                        self.model.apply(lambda m: setattr(m, 'width_mult', w))
                        # Execute sub-network at current width, yˆ = M0(x).
                        preds_w = self.model(images)
                        # Compute loss, loss = criterion(ˆy, y).
                        loss_w = self.loss_fn(preds_w, masks)
                        total_loss += loss_w
                        # Compute gradients, loss.backward().
                        self.scaler.scale(loss_w).backward()

                        # log per width loss
                        if config.use_tb and self.main_rank:
                            self.writer.add_scalar(f'train/loss{w}', loss_w.detach(), self.train_itrs)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema_model.update(self.model, self.train_itrs)

            loss = total_loss / len(config.slim_width_mult_list)
            if self.main_rank:
                pbar.set_description(('%s' * 2) %
                                     (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" " * 4}|',
                                      f'Loss:{loss.detach():4.4g}{" " * 4}|',)
                                     )

        return

    @torch.no_grad()
    def validate(self, config, loader, val_best=False):
        pbar = tqdm(loader) if self.main_rank else loader

        max_width = max(config.slim_width_mult_list)
        width_scores = {}
        for w in config.slim_width_mult_list:
            # change width
            self.ema_model.apply(lambda m: setattr(m, 'width_mult', w))
            # compute metrics for this width
            scores = self.val_compute_metrics(config, pbar)
            width_scores[w] = scores
            for metric in self.metrics:
                metric.reset()

        # if this is the main process (main_rank) - bc we are doing distributed training
        if self.main_rank:
            log_print = ""
            if val_best:
                log_print += f"\n\n----- Finished training {config.total_epoch} epochs -----"
            else:
                log_print += f"\n\n----- Trained {self.cur_epoch} epochs -----"

            for w in config.slim_width_mult_list:
                log_print += f"\n[Width: {w:.3f}] "

                for i in range(len(config.metrics)):
                    if val_best:
                        log_print += f"Final m{config.metrics[i]}={width_scores[w][i].mean():.4f} | "
                    else:
                        log_print += f"Current m{config.metrics[i]}={width_scores[w][i].mean():.4f} "
                        # best_score is the first metric of the largest width
                        if i == 0 and w == max_width:
                            log_print += f"[Best m{config.metrics[0]}={self.best_score:.4f}] "
                        log_print += f"| "

                    # if this is the last iteration trim the "| "
                    if i == len(config.metrics) - 1:
                        log_print = log_print[:-2]

                    if config.use_tb and self.cur_epoch < config.total_epoch:
                        self.writer.add_scalar(f'val/{w}/m{config.metrics[i]}', width_scores[w][i].mean().item(), self.cur_epoch + 1)
                        if config.metrics[i] == 'iou':
                            for j in range(config.num_class):
                                self.writer.add_scalar(f'val/{w}/IoU_cls{j:02f}', width_scores[w][i][j].item(), self.cur_epoch + 1)

            self.logger.info(log_print)

        return width_scores[max_width][0].mean()
