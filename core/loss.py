import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


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


class ModdedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6,
                 ignore_index=255):  # DUCKNet uses smoothness of 1e-6. added class_weights to handle disappearing channels
        """
        Dice Loss for segmentation tasks.
        :param smooth: Smoothing factor to avoid division by zero.
        :param ignore_index: Class index to be ignored in the loss computation.
        """
        super(ModdedDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.call_counter = 0  # DEBUG
        # self.class_weights = class_weights #weights for each class.

    def forward(self, logits, targets, save_path=None, class_weights=None):
        """
        Forward pass for Dice Loss computation.
        :param logits: Predicted logits (batch_size, num_classes, H, W).
        :param targets: Ground truth labels (batch_size, H, W).
        :return: Dice loss value.
        """
        num_classes = logits.shape[1]
        # Don't print number of classes, that's clear already.
        # print(f'The Number of Classes: {num_classes}')

        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)

        # One-hot encode the targets
        # targets_one_hot = torch.zeros_like(probs)
        # targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        ##print('shapes',logits.shape, targets.shape)
        # Targets are currently [batch, len, wid, channel]
        # Reorder target shapes to be [batch, channel, len, wid]

        targets = torch.permute(targets, (0, 3, 1, 2))

        '''
        #targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        # Ignore the ignore_index in loss calculation
        mask = targets != self.ignore_index
        probs = probs * mask.unsqueeze(1)  # Apply mask to ignore_index
        targets_one_hot = targets_one_hot * mask.unsqueeze(1)

        # Compute Dice coefficient
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff.mean()  # Taking mean across batch and classes
        '''
        # turn the logits into probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)  # Softmaxing along the channel dimensiomn

        # implement the DICE coefficient
        total_loss = 0.0
        counted_channels = 0
        # Convert probs to 1hot prediction
        pred_classes = probs.argmax(dim=1)
        pred_one_hot = torch.nn.functional.one_hot(pred_classes)
        # print(pred_onehot.shape) [B, H, W, 13]
        '''
        pred_one_hot = pred_one_hot.permute(0, 3, 1, 2)          # [B, 13, H, W]
        num_channels = probs.shape[1]
        '''
        prevent_0_crash = 1e-6  # No divide by 0
        pred_one_hot = probs  # Just use softmaxed predictions directly

        ##########################DEBUG################################i
        if save_path is not None:
            self.call_counter += 1  # Save every 10 loops.
            if self.call_counter > 100:
                debug_saver(probs, targets, save_path)
                self.call_counter = 0
        ###############################################################

        # for loop to discourage class dissapearance with training generated by chatgpt.
        for channel in range(probs.shape[1]):  # Over each channel
            labels = targets[:, channel, :, :].float()
            pred = probs[:, channel, :, :]  # Differentiable
            if labels.sum() == 0:
                continue  # Skip contribution to DICE if not class is not present

            # Compute DICE
            intersect = (pred * labels).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))
            dice_score = (2. * intersect + prevent_0_crash) / (union + prevent_0_crash)
            dice_loss = 1 - dice_score.mean()

            # Apply optional class weight
            # print(class_weights,'WEIGHTS')

            if class_weights is not None:
                dice_loss *= class_weights[channel]

            total_loss += dice_loss
            counted_channels += 1

        total_loss = total_loss / counted_channels  # Normalize DICE across channels now, since there's a loss for every channel
        return total_loss


class CEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255):  # DUCKNet uses smoothness of 1e-6
        """
        Dice Loss for segmentation tasks.
        :param smooth: Smoothing factor to avoid division by zero.
        :param ignore_index: Class index to be ignored in the loss computation.
        """
        super(CEDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.call_counter = 0  # DEBUG

    def forward(self, logits, targets, save_path=None, ce_loss_weight=1, class_weights=None):
        """
        Forward pass for Dice Loss computation.
        :param logits: Predicted logits (batch_size, num_classes, H, W).
        :param targets: Ground truth labels (batch_size, H, W).
        :return: Dice loss value.
        """
        num_classes = logits.shape[1]
        # Don't print number of classes, that's clear already.
        # print(f'The Number of Classes: {num_classes}')

        # Convert logits to probabilities using softmax
        # probs = F.softmax(logits, dim=1)

        # One-hot encode the targets
        # targets_one_hot = torch.zeros_like(probs)
        # targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        ##print('shapes',logits.shape, targets.shape)
        # Targets are currently [batch, len, wid, channel]
        # Reorder target shapes to be [batch, channel, len, wid]

        targets = torch.permute(targets, (0, 3, 1, 2))

        '''
        #targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        # Ignore the ignore_index in loss calculation
        mask = targets != self.ignore_index
        probs = probs * mask.unsqueeze(1)  # Apply mask to ignore_index
        targets_one_hot = targets_one_hot * mask.unsqueeze(1)

        # Compute Dice coefficient
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff.mean()  # Taking mean across batch and classes
        '''
        # Cut OUT the "other" channel
        probs = torch.nn.functional.softmax(logits, dim=1)  # Softmaxing along the channel dimensiomn

        # implement the DICE coefficient
        total_loss = 0.0
        counted_channels = 0

        # Convert probs to 1hot prediction
        pred_classes = probs.argmax(dim=1)
        pred_one_hot = torch.nn.functional.one_hot(pred_classes)
        # print(pred_onehot.shape) [B, H, W, 13]
        '''
        pred_one_hot = pred_one_hot.permute(0, 3, 1, 2)          # [B, 13, H, W]
        num_channels = probs.shape[1]
        '''
        prevent_0_crash = 1e-6  # No divide by 0
        pred_one_hot = probs  # Just use softmaxed predictions directly

        ##########################DEBUG################################i
        if save_path is not None:
            # if self.call_counter == 250:
            #    debug_saver(probs, targets, save_path, predictions_name = 'initialization')
            #    raise Exception('Stopping on initialization')
            self.call_counter += 1  # Save every 10 loops.
            # DEBUG
            if self.call_counter > 100:
                debug_saver(probs, targets, save_path)
                self.call_counter = 0

        ###############################################################

        # for loop to discourage class dissapearance with training generated by chatgpt.
        for channel in range(probs.shape[1]):  # Over each channel
            labels = targets[:, channel, :, :].float()
            pred = probs[:, channel, :, :]  # Differentiable
            if labels.sum() == 0:
                continue  # Skip contribution to DICE if not class is not present

            # Compute DICE
            intersect = (pred * labels).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + labels.sum(dim=(1, 2))
            dice_score = (2. * intersect + prevent_0_crash) / (union + prevent_0_crash)
            dice_loss = 1 - dice_score.mean()

            # Apply optional class weight
            # print(class_weights,'WEIGHTS')

            if class_weights is not None:
                dice_loss *= class_weights[channel]

            total_loss += dice_loss
            counted_channels += 1

        total_loss = total_loss / counted_channels  # Normalize DICE across channels now, since there's a loss for every channel

        class_label_targets = targets.argmax(dim=1).long()  # shape: [B, H, W]
        ce_loss = F.cross_entropy(logits, class_label_targets, ignore_index=self.ignore_index)

        # print(total_loss, ce_loss, "DICE, Crossentropy")
        total_loss += ce_loss  # taking the log decreases large contributions but encourages the lowering of the CE loss.
        return total_loss


class Channel1Loss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255):
        super(Channel1Loss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.call_counter = 0

    def forward(self, logits, targets, class_weights=None, save_path=None):
        """
        logits: [B, C, H, W]
        targets: [B, H, W, C] → will be permuted to [B, C, H, W]
        Only computes loss for class 1 (RA-RAF)
        """
        probs = F.softmax(logits, dim=1)
        targets = torch.permute(targets, (0, 3, 1, 2))  # [B, C, H, W]

        pred = probs[:, 1, :, :]  # Class 1
        label = targets[:, 1, :, :].float()

        ##########################DEBUG################################i
        if save_path is not None:
            # if self.call_counter == 250:
            #    debug_saver(probs, targets, save_path, predictions_name = 'initialization')
            #    raise Exception('Stopping on initialization')
            self.call_counter += 1  # Save every 10 loops.
            # DEBUG
            if self.call_counter > 100:
                debug_saver(probs, targets, save_path)
                self.call_counter = 0

        ###############################################################

        # Skip if no class 1 present
        if label.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        intersect = (pred * label).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + label.sum(dim=(1, 2))
        dice_score = (2. * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        # Cross-entropy only for class 1
        target_labels = targets.argmax(dim=1)  # [B, H, W]
        class_w = torch.zeros(logits.shape[1], device=logits.device)
        class_w[1] = 1.0
        ce_loss = F.cross_entropy(logits, target_labels, weight=class_w)

        return dice_loss + ce_loss

def debug_saver(probs, targets,save_path, predictions_name = 'predictions', targets_name = 'targets'):
    '''a debugger to save masks'''
    os.makedirs("debug_dice", exist_ok=True)

    # 1. Get the predicted class for each pixel: shape [B, H, W]
    pred_classes = torch.argmax(probs, dim=1)

    # 2. Convert to one-hot: shape [B, H, W, C]
    pred_one_hot = F.one_hot(pred_classes, num_classes=probs.shape[1])

    # 3. Permute to match shape [B, C, H, W] for Dice
    pred_one_hot = pred_one_hot.permute(0, 3, 1, 2).float()

    # 4. Save the binary one-hot predictions
    np.save(rf"{save_path}/{predictions_name}.npy", pred_one_hot.detach().cpu().numpy())

    # Optionally also save target and/or probs for comparison
    np.save(rf"{save_path}/{targets_name}.npy", targets.detach().cpu().numpy())
    print('Saved examples')

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

    elif config.loss_type == 'modded_dice':
        criterion = ModdedDiceLoss(ignore_index=config.ignore_index)

    elif config.loss_type == 'ce_dice':
        criterion = CEDiceLoss(ignore_index=config.ignore_index)

    #Debug Losses
    elif config.loss_type=='channel1loss':
        criterion = Channel1Loss(ignore_index=config.ignore_index)
    
    else:
        raise NotImplementedError(f"Unsupport loss type: {config.loss_type}")

    return criterion
