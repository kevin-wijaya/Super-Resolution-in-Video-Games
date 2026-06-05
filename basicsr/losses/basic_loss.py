import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import pywt

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss

@LOSS_REGISTRY.register()
class WaveletLoss(nn.Module):
    """
    Wavelet Loss.

    Compute L1 loss on wavelet subbands:
        LL, LH, HL, HH

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): 'mean' | 'sum'. Default: 'mean'.
        wavelet (str): Wavelet type. Default: 'haar'.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 wavelet='haar',
                 loss_weight_ll=1,
                 loss_weight_lh=1,
                 loss_weight_hl=1,
                 loss_weight_hh=1):
        super(WaveletLoss, self).__init__()

        if reduction not in ['mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. '
                f'Supported ones are: mean | sum'
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.wavelet = wavelet
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh

    def dwt2(self, x):
        """
        Args:
            x: Tensor (N, C, H, W)

        Returns:
            LL, LH, HL, HH
        """
        LL_list, LH_list, HL_list, HH_list = [], [], [], []

        for c in range(x.shape[1]):
            xc = x[:, c:c+1, :, :]

            ll_batch = []
            lh_batch = []
            hl_batch = []
            hh_batch = []

            for b in range(xc.shape[0]):
                arr = xc[b, 0].detach().cpu().numpy()

                LL, (LH, HL, HH) = pywt.dwt2(arr, self.wavelet)

                ll_batch.append(torch.tensor(LL))
                lh_batch.append(torch.tensor(LH))
                hl_batch.append(torch.tensor(HL))
                hh_batch.append(torch.tensor(HH))

            LL_list.append(torch.stack(ll_batch))
            LH_list.append(torch.stack(lh_batch))
            HL_list.append(torch.stack(hl_batch))
            HH_list.append(torch.stack(hh_batch))

        LL = torch.stack(LL_list, dim=1).to(x.device)
        LH = torch.stack(LH_list, dim=1).to(x.device)
        HL = torch.stack(HL_list, dim=1).to(x.device)
        HH = torch.stack(HH_list, dim=1).to(x.device)

        return LL, LH, HL, HH

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): (N, C, H, W)
            target (Tensor): (N, C, H, W)
        """

        pred_LL, pred_LH, pred_HL, pred_HH = self.dwt2(pred)
        gt_LL, gt_LH, gt_HL, gt_HH = self.dwt2(target)

        loss_LL = F.l1_loss(pred_LL, gt_LL, reduction=self.reduction)
        loss_LH = F.l1_loss(pred_LH, gt_LH, reduction=self.reduction)
        loss_HL = F.l1_loss(pred_HL, gt_HL, reduction=self.reduction)
        loss_HH = F.l1_loss(pred_HH, gt_HH, reduction=self.reduction)

        loss = (
            self.loss_weight_ll * loss_LL
            + self.loss_weight_lh * loss_LH
            + self.loss_weight_hl * loss_HL
            + self.loss_weight_hh * loss_HH
        )

        return self.loss_weight * loss