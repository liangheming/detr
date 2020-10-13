import torch

from torch import nn
from nets import resnet
from nets.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
import torch.nn.functional as F


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackBone(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, dilation=False):
        super(BackBone, self).__init__()
        self.backbone = getattr(resnet, backbone)(replace_stride_with_dilation=[False, False, dilation],
                                                  pretrained=pretrained)
        self.position_encoding = PositionEmbeddingSine(normalize=True)

    def forward(self, tensor_list):
        tensor_list.tensors = self.backbone(tensor_list.tensors)
        tensor_list.masks = F.interpolate(tensor_list.masks[None].float(),
                                          size=tensor_list.tensors.shape[-2:]).to(torch.bool)[0]
        pos = self.position_encoding(tensor_list.tensors)
        return tensor_list, pos


class DETR(nn.Module):
    def __init__(self):
        super(DETR, self).__init__()


