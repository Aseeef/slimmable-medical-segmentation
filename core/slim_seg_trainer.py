import random

import torch
from torch.cuda import amp
from tqdm import tqdm
from typing_extensions import override

from configs.base_config import SlimmableTrainingType
from core import SegTrainer
from core.loss import kd_loss_fn
from core.trainer_registry import register_trainer
from models.slimmable.slimmable_ops import bn_calibration_init
from utils import sampler_set_epoch

@register_trainer
class SlimmableSegTrainer(SegTrainer):

    is_calibrated: bool = False

    def __init__(self, config):
        super().__init__(config)

    @override
    def train_one_epoch(self, config):
        if config.slimmable_training_type == SlimmableTrainingType.NONE.value:
            return super().train_one_epoch(config)

        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch)

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (images, masks) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()

            total_loss = 0.0
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

            # NOTE: Many of my comments here are copied directly from the slim-net and us-net papers
            else:
                if config.slimmable_training_type == SlimmableTrainingType.S_NET.value:
                    with amp.autocast(enabled=config.amp_training):
                        widths = sorted(config.slim_width_mult_list, reverse=True)
                        # train from highest to lowest
                        for w in widths:
                            # Switch the batch normalization parameters of current width on network M.
                            self.model.apply(lambda m: setattr(m, 'width_mult', w))
                            # Execute subnetwork at current width, yˆ = M0(x).
                            preds_w = self.model(images)
                            # Compute loss, loss = criterion(ˆy, y).
                            loss_w = self.loss_fn(preds_w, masks)
                            total_loss += loss_w
                            # Compute gradients, loss.backward().
                            self.scaler.scale(loss_w).backward()
                            # log per width loss
                            if config.use_tb and self.main_rank:
                                self.writer.add_scalar(f'train/loss{w}', loss_w.detach(), self.train_itrs)

                elif config.slimmable_training_type == SlimmableTrainingType.US_NET.value:
                    assert len(config.slim_width_range) == 2, \
                        "US-Net requires 2 widths, the min-width and the max-width inside slim_width_range."
                    assert config.slim_width_range[0] < config.slim_width_range[1], \
                        "slim_width_range[0] must be less than slim_width_range[1]."

                    min_width = config.slim_width_range[0]
                    max_width = config.slim_width_range[1]

                    # always train smallest + largest widths as per the sandwich rule (see paper)
                    widths_train = [max_width, min_width]
                    # Randomly sample (n−2) widths, as width samples.
                    for _ in range(config.us_num_training_samples - 2):
                        widths_train.append(random.uniform(min_width, max_width))

                    for w in sorted(widths_train, reverse=True):
                        # Note: paper also described something called
                        #  "non-uniform training" but honestly, its not worth getting into for us
                        self.model.apply(lambda m: setattr(m, 'width_mult', w))

                        if config.inplace_distillation:
                            # Execute subnetwork at current width, yˆ = M0(x).
                            if w == max_width:
                                # Execute full-network, y′ = M(x)
                                soft_masks = self.model(images)
                                # Compute loss, loss = criterion(y′, y).
                                # (loss_fn is dice loss)
                                loss_w = self.loss_fn(soft_masks, masks)
                                total_loss += loss_w
                                # Accumulate gradients, loss.backward()
                                self.scaler.scale(loss_w).backward()
                                # Stop gradients of y′ as label, y′ = y′.detach().
                                soft_masks = soft_masks.detach()
                            else:
                                # Execute subnetwork at width, yˆ = M′(x).
                                preds_w = self.model(images)
                                # Compute loss, loss = criterion(yˆ, y′).
                                # since according to the paper, pure distillation is better than
                                # including ground truth, we train using the KL-Div
                                # Also note: this uses the KD formula proposed by Hinton et al. in the original KD paper
                                # (2015) for no other reason than this loss function came with our starter repo
                                loss_w = config.kd_loss_coefficient * kd_loss_fn(config, preds_w, soft_masks)
                                total_loss += loss_w
                                # Accumulate gradients, loss.backward()
                                self.scaler.scale(loss_w).backward()
                        # same as normal training
                        else:
                            # Switch the batch normalization parameters of current width on network M.
                            self.model.apply(lambda m: setattr(m, 'width_mult', w))
                            # Execute subnetwork at current width, yˆ = M0(x).
                            preds_w = self.model(images)
                            # Compute loss, loss = criterion(ˆy, y).
                            loss_w = self.loss_fn(preds_w, masks)
                            total_loss += loss_w
                            # Compute gradients, loss.backward().
                            self.scaler.scale(loss_w).backward()

                        # log per width loss for the min/max
                        if config.use_tb and self.main_rank and w in [min_width, max_width]:
                            self.writer.add_scalar(f'train/loss{w}', loss_w.detach(), self.train_itrs)

                else:
                    # illegal state
                    raise ValueError(f"Illegal state. Should not have training type: {config.slimmable_training_type}.")

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.ema_model.update(self.model, self.train_itrs)

            loss = total_loss / config.us_num_training_samples
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

    def calibrate_bn(self, config):
        """
        In order to make predictions for any width, we need to calibrate the batch norm statistic for that width first
        before we can use it to make predictions. This function does that.
        :param config:
        :return:
        """
        if config.slimmable_training_type == SlimmableTrainingType.US_NET.value:
            self.model.train()  # allows BN statistics to accumulate
            self.model.apply(bn_calibration_init)
            with torch.no_grad():  # Avoids gradient buildup
                i = 0
                for w in config.slim_width_mult_list:
                    self.model.apply(lambda m: setattr(m, 'width_mult', w))
                    for images, _ in self.train_loader:
                        self.model(images.to(self.device, dtype=torch.float32))
                        i += 1
                        if i >= config.bn_calibration_batch_size:
                            break
            self.model.eval()
            self.is_calibrated = True

    def predict(self, config, running_width=None):
        if not self.is_calibrated and config.slimmable_training_type == SlimmableTrainingType.US_NET.value:
            self.calibrate_bn(config)
        if running_width is None:
            running_width = max(config.slim_width_mult_list)
        if running_width not in config.slim_width_mult_list:
            raise ValueError(f"Invalid running_width: {running_width}. Must be one of {config.slim_width_mult_list}.")
        if len(self.train_loader) < config.bn_calibration_batch_size:
            self.logger.warning(f"Running width {running_width} is being used for prediction,"
                                f" but the training loader only has {len(self.test_loader)} batches. ")
        self.model.apply(lambda m: setattr(m, 'width_mult', running_width))
        super().predict(config)
