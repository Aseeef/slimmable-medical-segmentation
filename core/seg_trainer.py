import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from utils import (get_seg_metrics, sampler_set_epoch, get_colormap)
from .trainer_registry import register_trainer

from utils import de_parallel #For save_ckpt overwrite


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
                    #Visualize preds during training


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
            #print("[VALIDATION IMAGE] dtype:", images.dtype)
            #print("[VALIDATION IMAGE] min/max:", images.min().item(), images.max().item())
            #print("[VALIDATION IMAGE] shape:", images.shape)

            # In order to validate different size of images, we can resize the image to be compatible with the model's stride,
            # and then resize the predicted mask back to the image size to calculate the metric.
            # If you don't want this, simply set `val_img_stride=1` within the config file.
            _, _, H, W = images.shape
            stride = config.val_img_stride
            if H % stride != 0 or W % stride != 0:
                new_size = (H // stride * stride, W // stride * stride)
                images = F.interpolate(images, new_size, mode='bilinear')

            masks = masks.to(self.device, dtype=torch.long)
            preds = self.model(images) #change this to models if it breaks.

            ####################################
            if self.main_rank and not val_best:
                self.colormap = torch.tensor(get_colormap(config)).to(self.device)

                os.makedirs(os.path.join(config.save_dir, 'val_outputs'), exist_ok=True)

                # Convert preds to class index
                pred_classes = preds.argmax(dim=1)  # [B, H, W]
                pred_colors = self.colormap[pred_classes]  # [B, H, W, 3]
                pred_colors = pred_colors.cpu().numpy()  # numpy uint8

                # Get original images for blending
                input_images = images.cpu().numpy()

                for i in range(pred_colors.shape[0]):
                    pred_img = Image.fromarray(pred_colors[i].astype(np.uint8))
                    save_name = f"val_epoch{self.cur_epoch}_sample{i}.png"
                    save_path = os.path.join(config.save_dir, 'val_outputs', save_name)

                    if config.save_mask:
                        pred_img.convert("L").save(save_path)

                    if config.blend_prediction:
                        input_arr = input_images[i]
                        if input_arr.shape[0] == 1:  # Grayscale
                            input_arr = input_arr.squeeze(0)
                        elif input_arr.shape[0] == 3:  # RGB (C, H, W) -> (H, W, C)
                            input_arr = np.transpose(input_arr, (1, 2, 0))
                        input_arr = input_arr.astype(np.uint8)

                        input_img = Image.fromarray(input_arr)
                        blended = Image.blend(input_img.convert("RGBA"), pred_img.convert("RGBA"), alpha=config.blend_alpha)
                        blended.save(save_path.replace(".png", "_blend.png"))

                self.logger.info(f"[VAL] Saved predicted masks to {os.path.join(config.save_dir, 'val_outputs')}")
            ####################################

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

        self.load_ckpt(config)
        self.logger.info(f'[!] Loaded checkpoints from {config.load_ckpt_path} before predict')


        for (images, images_aug, img_names) in tqdm(self.test_loader):
            images_aug = images_aug.to(self.device, dtype=torch.float32)

            images = images.to(self.device, dtype=torch.float32)
            print("[TEST IMAGE] dtype:", images.dtype)
            print("[TEST IMAGE] min/max:", images.min().item(), images.max().item())
            print("[TEST IMAGE] shape:", images.shape)

            ####
            #CRITICAL ERROR: AUGMENTATION IS FAILING MISERABLY. USE THE NORMAL, UNAUGMENTED TEST SET!!!!
            ##IMAGES HAVE SHAPE [B H W C]
            #reshape the images being input
            images_passed_in = images.permute(0, 3, 1, 2)

            preds = self.model(images_passed_in)

            #Old code to visualize logit channels
            '''
            # Detach and move to CPU. Saving raw logits as matplotlib. 
            preds_np = preds.detach().cpu().squeeze(1).numpy()  # Shape: [B, H, W]
            import matplotlib.pyplot as plt
            for i in range(preds_np.shape[0]):
                diff_arr = preds_np[i,0]-preds_np[i,1]
                binary_mask = (diff_arr > 0).astype(np.uint8)
                plt.imshow(preds_np[i,1], cmap='viridis')  # or 'plasma' / 'gray'
                plt.colorbar()
                plt.title(f'Raw Logits - {img_names[i]}')
                save_path = os.path.join(config.save_dir, img_names[i] + '_logits.png')
                plt.savefig(save_path)
                plt.clf()  # clear figure for the next one
            '''


            preds = self.colormap[preds.max(dim=1)[1]].cpu().numpy()

            images = images.cpu().numpy()

            # Saving results
            for i in range(preds.shape[0]):
                save_path = os.path.join(config.save_dir, img_names[i])
                save_suffix = img_names[i].split('.')[-1]

                pred = Image.fromarray(preds[i].astype(np.uint8))

                if config.save_mask:
                    pred_save = pred.convert("L")
                    pred_save.save(save_path)


                if config.blend_prediction:
                    save_blend_path = save_path.replace(f'.{save_suffix}', f'_blend.{save_suffix}')

                    image = Image.fromarray(images[i].astype(np.uint8))
                    image = Image.blend(image, pred, config.blend_alpha)
                    image.save(save_blend_path)


    #OVERWRITE base trainer save_ckpt class. Saving NOT the EMA model, but the actual network model.
    def save_ckpt(self, config, save_best=False):
        if config.ckpt_name is None:
            save_name = 'best.pth' if save_best else 'last.pth'
        save_path = f'{config.save_dir}/{save_name}'
        state_dict = de_parallel(self.model).state_dict()
        torch.save({
            'cur_epoch': self.cur_epoch,
            'best_score': self.best_score,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict() if not save_best else None,
            'scheduler': self.scheduler.state_dict() if not save_best else None,
        }, save_path)

        if self.main_rank:
            self.logger.info(f"[CKPT] Saved checkpoint to {save_path}")