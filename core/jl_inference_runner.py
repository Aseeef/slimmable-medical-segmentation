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


class SlimmableSegInferenceRunner:
    """
    Standalone inference runner for slimmable segmentation models.
    Iterates over width multipliers to perform forward passes and collect predictions.
    """

    def __init__(self, model, val_loader: DataLoader, device: torch.device,
                 width_mult_list, amp_enabled=False, ddp=False, current_epoch=0, main_rank=True):
        """
        Args:
            model: The slimmable model with 'width_mult' attribute per layer/module.
            val_loader (DataLoader): Validation dataloader.
            device (torch.device): CUDA or CPU device.
            width_mult_list (list): List of width multipliers to evaluate.
            amp_enabled (bool): Whether to use mixed precision (torch.cuda.amp).
            ddp (bool): DistributedDataParallel flag.
            current_epoch (int): Current epoch number for reproducibility.
            main_rank (bool): Whether this is the main process (for tqdm display).
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.width_mult_list = width_mult_list
        self.amp_enabled = amp_enabled
        self.ddp = ddp
        self.current_epoch = current_epoch
        self.main_rank = main_rank

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