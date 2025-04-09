import torch
import torch.nn.functional as F
from torch.cuda import amp
from tqdm import tqdm

from .seg_trainer import SegTrainer
from models.slimmable.slimmable_ops import SwitchableBatchNorm2d

class SlimmableSegTrainer(SegTrainer):
    """
    A trainer for our slimmable DUCKNet implementation.
    Overrides train_one_epoch to repeatedly train the NN for each width multiplier in a single batch.
    """

    def __init__(self, config):
        super().__init__(config)

        # width_mult_list = [0.25, 0.5, 0.75, 1.0]
        self.width_mult_list = config.slim_width_mult_list  
        self.inplace_distill = getattr(config, 'inplace_distill', False)

    def train_one_epoch(self, config):
        """
        Slimmable training for one epoch. 
        Args: config set in projects/Team_A+/<top_dir_name>/configs

        """
        self.model.train()

        if config.DDP:
            self.train_loader.sampler.set_epoch(self.cur_epoch)

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (images, masks) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = images.to(self.device, dtype=torch.float32)
            masks  = masks.to(self.device, dtype=torch.long)
            self.optimizer.zero_grad()

            # First pass with the largest width
            largest_width = max(self.width_mult_list)
            self.model.apply(
                lambda m: setattr(m, 'width_mult', largest_width)
            )

            with amp.autocast(enabled=config.amp_training):                         # Forward begins
                preds_largest = self.model(images)
                loss_largest = self.loss_fn(preds_largest, masks)
                
                # Use other teacher outputs if KD is set
                if config.kd_training:
                    with torch.no_grad():
                        teacher_preds = self.teacher_model(images)
                    loss_kd = self._kd_loss(config, preds_largest, teacher_preds)
                    loss_largest += config.kd_loss_coefficient * loss_kd

            # For in-place distillation
            if self.inplace_distill:
                with torch.no_grad():
                    soft_target = F.softmax(preds_largest, dim=1)   # save the soft target/largest pred
            else:
                soft_target = None

            self.scaler.scale(loss_largest).backward(retain_graph=self.inplace_distill) # backprop on the largest

            # Second pass on other widths
            for w in sorted(self.width_mult_list, reverse=True):
                if w == largest_width:
                    continue

                self.model.apply(lambda m: setattr(m, 'width_mult', w))

                with amp.autocast(enabled=config.amp_training):
                    preds = self.model(images)
                    loss_w = self.loss_fn(preds, masks)

                    # checking if enabled in the config KD from the teacher
                    if config.kd_training:
                        with torch.no_grad():
                            teacher_preds = self.teacher_model(images)
                        loss_kd_w = self._kd_loss(config, preds, teacher_preds)
                        loss_w += config.kd_loss_coefficient * loss_kd_w

                    # In place distillation from largest predictions
                    if self.inplace_distill and (soft_target is not None):
                        loss_inplace_kd = F.kl_div(
                            input=F.log_softmax(preds, dim=1), 
                            target=soft_target, 
                            reduction='batchmean'
                        )
                        loss_w += config.inplace_distill_coef * loss_inplace_kd     # adjusting the coefficient

                # backprop step for smaller width
                self.scaler.scale(loss_w).backward()

            # Optimizer update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()
            self.ema_model.update(self.model, self.train_itrs)

            # pbar and logging
            if self.main_rank:
                total_loss_display = loss_largest.detach().item()
                pbar.set_description((
                    f"Epoch:{self.cur_epoch}/{config.total_epoch} "
                    f"| Loss:{total_loss_display:4.4g} "
                ))

            # if use_tb enabled in base_config
            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss_largest', loss_largest.detach(), self.train_itrs)

        # next iter for the next mini batch

    def _kd_loss(self, config, student_preds, teacher_preds):
        """
        Basic KD term. THIS MAY NEED TO BE CHANGED.
        Does a KL div with teacher’s logits.
        """
        # log-softmax + softmax KL
        T = config.kd_temp
        student_log_probs = F.log_softmax(student_preds / T, dim=1)
        teacher_probs     = F.softmax(teacher_preds / T,    dim=1)
        loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return loss_kl * (T * T)  # classical factor for temperature scaling
