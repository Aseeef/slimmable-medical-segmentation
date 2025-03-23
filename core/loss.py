import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_index=255):
        super().__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)


# This loss function was generated with AI help
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255):  # DUCKNet uses smoothness of 1e-6
        """
        Dice Loss for segmentation tasks.
        :param smooth: Smoothing factor to avoid division by zero.
        :param ignore_index: Class index to be ignored in the loss computation.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Forward pass for Dice Loss computation.
        :param logits: Predicted logits (batch_size, num_classes, H, W).
        :param targets: Ground truth labels (batch_size, H, W).
        :return: Dice loss value.
        """
        # Ensure targets are a PyTorch tensor and long type
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.long, device=logits.device)
        else:
            targets = targets.to(dtype=torch.long)

        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)

        # One-hot encode the targets
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Ignore the ignore_index in loss calculation
        mask = targets != self.ignore_index
        probs = probs * mask.unsqueeze(1)  # Apply mask to ignore_index
        targets_one_hot = targets_one_hot * mask.unsqueeze(1)

        # Compute Dice coefficient
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff.mean()  # Taking mean across batch and classes

        return dice_loss



def get_loss_fn(config, device):
    if config.class_weights is None:
        weights = None
    else:
        weights = torch.Tensor(config.class_weights).to(device)

    if config.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index, 
                                        reduction=config.reduction, weight=weights)

    elif config.loss_type == 'ohem':
        criterion = OhemCELoss(thresh=config.ohem_thrs, ignore_index=config.ignore_index)
        
    elif config.loss_type == 'dice':
        criterion = DiceLoss(ignore_index=config.ignore_index)

    else:
        raise NotImplementedError(f"Unsupport loss type: {config.loss_type}")

    return criterion


def kd_loss_fn(config, outputs, outputsT):
    if config.kd_loss_type == 'kl_div':
        lossT = F.kl_div(F.log_softmax(outputs/config.kd_temperature, dim=1),
                    F.softmax(outputsT.detach()/config.kd_temperature, dim=1)) * config.kd_temperature ** 2

    elif config.kd_loss_type == 'mse':
        lossT = F.mse_loss(outputs, outputsT.detach())

    return lossT