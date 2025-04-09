import random
from enum import Enum

import torch
from torch import amp
import torch.nn.functional as F
from tqdm import tqdm
from typing_extensions import override

from core import SegTrainer
from core.trainer_registry import register_trainer
from utils import sampler_set_epoch


class SlimmableTrainingType(Enum):
    NONE = "none"
    S_NET = "s-net"
    US_NET = "us-net"

@register_trainer
class SlimmableSegTrainer(SegTrainer):
    def __init__(self, config):
        super().__init__(config)
        # TODO: special training logic required
        ...

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

            # change learning rate in each iteration
            if config.universally_slimmable_training:
                max_width = config.width_mult_range[1]
                min_width = config.width_mult_range[0]
            else:
                # s-nets
                max_width = max(config.width_mult_list)
                min_width = min(config.width_mult_list)

            # Forward path
            if config.slimmable_training:
                if config.universally_slimmable_training:
                    # universally slimmable model (us-nets)
                    # TODO
                    ...
                else:
                    # slimmable model (s-nets)
                    for width_mult in sorted(
                            config.width_mult_list, reverse=True):
                        self.model.apply(
                            lambda m: setattr(m, 'width_mult', width_mult))
                        if is_master():
                            meter = meters[str(width_mult)]
                        else:
                            meter = None
                        if width_mult == max_width:
                            loss, soft_target = self.loss_fn(
                                model, criterion, input, target, meter,
                                return_soft_target=True)
                        else:
                            if getattr(FLAGS, 'inplace_distill', False):
                                loss = self.loss_fn(
                                    model, criterion, input, target, meter,
                                    soft_target=soft_target.detach(),
                                    soft_criterion=soft_criterion)
                            else:
                                loss = self.loss_fn(preds, masks)
                        loss.backward()
            else:
                loss = forward_loss(
                    model, criterion, input, target, meters)
                loss.backward()

            # TODO: knowledge distillation?

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

        epoch, loader, model, criterion, optimizer, meters, phase = 'train',
        soft_criterion = None

        t_start = time.time()
        assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
        train = phase == 'train'
        if train:
            model.train()
        else:
            model.eval()
            if phase == 'cal':
                model.apply(bn_calibration_init)
        # change learning rate in each iteration
        if config.universally_slimmable_training:
            max_width = config.width_mult_range[1]
            min_width = config.width_mult_range[0]
        elif config.slimmable_training:
            max_width = max(config.width_mult_list)
            min_width = min(config.width_mult_list)

        if getattr(FLAGS, 'distributed', False):
            loader.sampler.set_epoch(epoch)
        for batch_idx, (input, target) in enumerate(loader):
            if phase == 'cal':
                if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
                    break
            target = target.cuda(non_blocking=True)
            if train:
                # change learning rate if necessary
                lr_schedule_per_iteration(optimizer, epoch, batch_idx)
                optimizer.zero_grad()
                if config.slimmable_training:
                    if config.universally_slimmable_training:
                        # universally slimmable model (us-nets)
                        widths_train = []
                        for _ in range(getattr(FLAGS, 'num_sample_training', 2) - 2):
                            widths_train.append(
                                random.uniform(min_width, max_width))
                        widths_train = [max_width, min_width] + widths_train
                        for width_mult in widths_train:
                            # the sandwich rule
                            if width_mult in [max_width, min_width]:
                                model.apply(
                                    lambda m: setattr(m, 'width_mult', width_mult))
                            elif getattr(FLAGS, 'nonuniform', False):
                                model.apply(lambda m: setattr(
                                    m, 'width_mult',
                                    lambda: random.uniform(min_width, max_width)))
                            else:
                                model.apply(lambda m: setattr(
                                    m, 'width_mult',
                                    width_mult))

                            # always track largest model and smallest model
                            if is_master() and width_mult in [
                                max_width, min_width]:
                                meter = meters[str(width_mult)]
                            else:
                                meter = None

                            # inplace distillation
                            if width_mult == max_width:
                                loss, soft_target = forward_loss(
                                    model, criterion, input, target, meter,
                                    return_soft_target=True)
                            else:
                                if getattr(FLAGS, 'inplace_distill', False):
                                    loss = forward_loss(
                                        model, criterion, input, target, meter,
                                        soft_target=soft_target.detach(),
                                        soft_criterion=soft_criterion)
                                else:
                                    loss = forward_loss(
                                        model, criterion, input, target, meter)
                            loss.backward()
                    else:
                        # slimmable model (s-nets)
                        for width_mult in sorted(
                                FLAGS.width_mult_list, reverse=True):
                            model.apply(
                                lambda m: setattr(m, 'width_mult', width_mult))
                            if is_master():
                                meter = meters[str(width_mult)]
                            else:
                                meter = None
                            if width_mult == max_width:
                                loss, soft_target = forward_loss(
                                    model, criterion, input, target, meter,
                                    return_soft_target=True)
                            else:
                                if getattr(FLAGS, 'inplace_distill', False):
                                    loss = forward_loss(
                                        model, criterion, input, target, meter,
                                        soft_target=soft_target.detach(),
                                        soft_criterion=soft_criterion)
                                else:
                                    loss = forward_loss(
                                        model, criterion, input, target, meter)
                            loss.backward()
                else:
                    loss = forward_loss(
                        model, criterion, input, target, meters)
                    loss.backward()
                if (getattr(FLAGS, 'distributed', False)
                        and getattr(FLAGS, 'distributed_all_reduce', False)):
                    allreduce_grads(model)
                optimizer.step()
                if is_master() and getattr(FLAGS, 'slimmable_training', False):
                    for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                        meter = meters[str(width_mult)]
                        meter['lr'].cache(optimizer.param_groups[0]['lr'])
                elif is_master():
                    meters['lr'].cache(optimizer.param_groups[0]['lr'])
                else:
                    pass
            else:
                if getattr(FLAGS, 'slimmable_training', False):
                    for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                        model.apply(
                            lambda m: setattr(m, 'width_mult', width_mult))
                        if is_master():
                            meter = meters[str(width_mult)]
                        else:
                            meter = None
                        forward_loss(model, criterion, input, target, meter)
                else:
                    forward_loss(model, criterion, input, target, meters)
        if is_master() and getattr(FLAGS, 'slimmable_training', False):
            for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
                results = flush_scalar_meters(meters[str(width_mult)])
                print('{:.1f}s\t{}\t{}\t{}/{}: '.format(
                    time.time() - t_start, phase, str(width_mult), epoch,
                    FLAGS.num_epochs) + ', '.join(
                    '{}: {:.3f}'.format(k, v) for k, v in results.items()))
        elif is_master():
            results = flush_scalar_meters(meters)
            print(
                '{:.1f}s\t{}\t{}/{}: '.format(
                    time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
                ', '.join('{}: {:.3f}'.format(k, v) for k, v in results.items()))
        else:
            results = None
        return results