import torch

from torch import nn
from nets import resnet
from nets.position_encoding import PositionEmbeddingSine
from nets.transformer import Transformer
from commons.boxs_utils import xywh2xyxy
from losses.detr_loss import DETRLoss
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
    def __init__(self,
                 d_model=512,
                 backbone="resnet18",
                 pretrained=True,
                 dilation=False,
                 frozen_bn=True,
                 trainable=True):
        super(BackBone, self).__init__()
        if frozen_bn:
            backbone = getattr(resnet, backbone)(replace_stride_with_dilation=[False, False, dilation],
                                                 pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(resnet, backbone)(replace_stride_with_dilation=[False, False, dilation],
                                                 pretrained=pretrained)
        for name, parameter in backbone.named_parameters():
            if not trainable or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        self.backbone = backbone
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
        self.last_channels = self.backbone.last_channels

    def forward(self, tensor_list):
        tensor_list.tensors = self.backbone(tensor_list.tensors)
        tensor_list.masks = F.interpolate(tensor_list.masks[None].float(),
                                          size=tensor_list.tensors.shape[-2:]).to(torch.bool)[0]
        pos = self.position_encoding(tensor_list.tensors, mask=tensor_list.masks)
        return tensor_list, pos


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


default_config = {
    "num_cls": 80,
    "num_queries": 100,
    "aux_loss": True,
    # backbone
    "backbone": "resnet50",
    "pretrained": True,
    "dilation": False,
    "frozen_bn": True,
    "trainable": True,
    # transformer
    "d_model": 256,
    "n_head": 8,
    "num_encoder_layer": 6,
    "num_decoder_layer": 6,
    "dim_feedforward": 2048,
    "drop_out": 0.1,
    "normalize_before": False,
    "return_intermediate_dec": True,
    # loss
    "eos_coef": 0.1,
    "cls_loss_weights": 1.0,
    "dis_loss_weights": 5.0,
    "iou_loss_weights": 2.0,
    "cls_sim_weights": 1.0,
    "dis_sim_weights": 5.0,
    "iou_sim_weights": 2.0,
    "score_thresh": 0.01
}


class DETR(nn.Module):
    def __init__(self, cfg=None):
        super(DETR, self).__init__()
        if cfg is None:
            cfg = default_config
        else:
            cfg = {**default_config, **cfg}
        self.score_thresh = cfg['score_thresh']
        self.transformer = Transformer(
            d_model=cfg['d_model'],
            n_head=cfg['n_head'],
            num_encoder_layer=cfg['num_encoder_layer'],
            num_decoder_layer=cfg['num_decoder_layer'],
            dim_feedforward=cfg['dim_feedforward'],
            drop_out=cfg['drop_out'],
            normalize_before=cfg['normalize_before'],
            return_intermediate_dec=cfg['return_intermediate_dec']
        )
        hidden_dim = self.transformer.d_model
        self.backbone = BackBone(
            d_model=hidden_dim,
            backbone=cfg['backbone'],
            pretrained=cfg['pretrained'],
            frozen_bn=cfg['frozen_bn'],
            trainable=cfg['trainable'],
            dilation=cfg['dilation']
        )
        self.class_embed = nn.Linear(hidden_dim, cfg['num_cls'] + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(cfg['num_queries'], hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.last_channels, hidden_dim, kernel_size=1)
        self.aux_loss = cfg['aux_loss']
        self.creterion = DETRLoss(
            num_cls=cfg['num_cls'],
            eos_coef=cfg['eos_coef'],
            cls_loss_weights=cfg['cls_loss_weights'],
            dis_loss_weights=cfg['dis_loss_weights'],
            iou_loss_weights=cfg['iou_loss_weights'],
            cls_sim_weights=cfg['cls_sim_weights'],
            dis_sim_weights=cfg['dis_sim_weights'],
            iou_sim_weights=cfg['iou_sim_weights']
        )

    def forward(self, tensor_list):
        features, pos = self.backbone(tensor_list)
        src, mask = features.tensors, features.masks
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        loss = dict()
        predicts = None
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.training:
            cls_loss, dis_loss, iou_loss = self.creterion(out, [{"labels": label, "boxes": box} for label, box in
                                                                zip(tensor_list.labels, tensor_list.boxes)])
            loss['cls_loss'] = cls_loss
            loss['dis_loss'] = dis_loss
            loss['iou_loss'] = iou_loss
        else:
            predicts = self.post_process(out, tensor_list.batch_shape)
        return predicts, loss

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def post_process(self, outputs, targets_shape):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = out_logits.float().softmax(dim=-1)
        scores, labels = prob[..., :-1].max(-1)
        boxes = out_bbox.float().sigmoid()
        boxes = xywh2xyxy(boxes)
        boxes[..., [0, 2]] = boxes[..., [0, 2]] * targets_shape[0]
        boxes[..., [1, 3]] = boxes[..., [1, 3]] * targets_shape[1]
        ret = torch.cat([boxes, scores.unsqueeze(-1), labels.float().unsqueeze(-1)], dim=-1)
        ret_list = list()
        for item in ret:
            valid_mask = (item[:, -2] > self.score_thresh)
            if valid_mask.sum().item() > 0:
                ret_list.append(item[valid_mask, :])
            else:
                ret_list.append(None)
        return ret_list
