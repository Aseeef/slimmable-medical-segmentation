'''
    Purpose: 
    inference runner for slimmable Ducknet
'''

import torch
from torch.cuda import amp
from tqdm import tqdm

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

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




@register_trainer
class SlimmableSegInferenceRunner(BaseTrainer):
    """
    Standalone inference runner for slimmable segmentation models.
    Iterates over width multipliers to perform forward passes and collect predictions.
    """

    def __init__(self, config):
        super().__init__(config)
        if config.is_testing:
            self.colormap = torch.tensor(get_colormap(config)).to(self.device)
        else:
            self.metrics = [get_seg_metrics(config, metric_name).to(self.device) for metric_name in config.metrics]

            
    @torch.no_grad()
    def run(self):
        """
        Perform inference for all width multipliers.
        Returns:
            predictions_dict: {width_multiplier: list of output tensors}
        """
        self.model.eval()
        predictions_dict = {w: [] for w in self.width_mult_list}

        if self.ddp and hasattr(self.val_loader.sampler, 'set_epoch'):
            self.val_loader.sampler.set_epoch(self.current_epoch)

        data_iter = tqdm(self.val_loader, desc="Running Inference") if self.main_rank else self.val_loader

        for images, _ in data_iter:
            images = images.to(self.device, dtype=torch.float32)

            for w in sorted(self.width_mult_list, reverse=True):
                self._set_width(w)

                with amp.autocast(enabled=self.amp_enabled):
                    outputs = self.model(images)

                predictions_dict[w].append(outputs.cpu())

                if self.main_rank:
                    data_iter.set_description(f"Inference | Width {w}")

        # Concatenate outputs for each width
        predictions_dict = {
            w: torch.cat(pred_list, dim=0) for w, pred_list in predictions_dict.items()
        }

        return predictions_dict

    def _set_width(self, width):
        """
        Sets the width multiplier on all submodules of the model that have a 'width_mult' attribute.
        """
        def set_attr_fn(m):
            if hasattr(m, 'width_mult'):
                m.width_mult = width
        self.model.apply(set_attr_fn)