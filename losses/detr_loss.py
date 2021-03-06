import torch
from torch import nn
# from losses.commons import BoxSimilarity
from commons.boxs_utils import xyxy2xywh, xywh2xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment

import torch.nn.functional as f


class DETRLoss(nn.Module):
    def __init__(self,
                 num_cls=80,
                 eos_coef=0.1,
                 cls_loss_weights=1.0,
                 dis_loss_weights=5.0,
                 iou_loss_weights=2.0,
                 cls_sim_weights=1.0,
                 dis_sim_weights=5.0,
                 iou_sim_weights=2.0):
        super(DETRLoss, self).__init__()
        self.num_cls = num_cls
        self.matcher = HungarianMatcher(cls_weights=cls_sim_weights,
                                        distance_weights=dis_sim_weights,
                                        iou_weights=iou_sim_weights)
        self.iou_loss = generalized_box_iou
        empty_weight = torch.ones(self.num_cls + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.loss_weights_dict = {"cls_loss": cls_loss_weights,
                                  "dis_loss": dis_loss_weights,
                                  "iou_loss": iou_loss_weights}

    def get_loss(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes'].float().sigmoid()
        tgt_logits_tensor = torch.full(size=pred_logits.shape[:-1],
                                       fill_value=self.num_cls,
                                       dtype=torch.int64,
                                       device=pred_logits.device)
        pred_boxes_list = list()
        tgt_boxes_list = list()
        for bid, (pred_logits_item, pred_boxes_item, tgt_item, idx) in \
                enumerate(zip(pred_logits, pred_boxes, targets, indices)):
            tgt_labels_item, tgt_boxes_item = tgt_item['labels'], tgt_item['boxes']
            i, j = idx
            tgt_logits_tensor[bid, i] = tgt_labels_item[j].long()
            pred_boxes_list.append(pred_boxes_item[i])
            tgt_boxes_list.append(tgt_boxes_item[j])
        pred_boxes_tensor = torch.cat(pred_boxes_list, dim=0)
        tgt_boxes_tensor = torch.cat(tgt_boxes_list, dim=0)
        ce_loss = f.cross_entropy(pred_logits.transpose(1, 2), tgt_logits_tensor, self.empty_weight)
        distance_loss = f.l1_loss(pred_boxes_tensor, xyxy2xywh(tgt_boxes_tensor), reduction="none").sum(-1).mean()
        iou_loss = (1 - torch.diag(self.iou_loss(xywh2xyxy(pred_boxes_tensor), tgt_boxes_tensor))).mean()
        return {"cls_loss": ce_loss, "dis_loss": distance_loss, "iou_loss": iou_loss}

    def forward(self, outputs, targets):
        loss_list = list()
        loss_list.append(self.get_loss(outputs, targets))
        aux_outputs = outputs['aux_outputs']
        if len(aux_outputs):
            for layer_out in aux_outputs:
                loss_list.append(self.get_loss(layer_out, targets))
        cls_loss = list()
        dis_loss = list()
        iou_loss = list()
        for loss_item in loss_list:
            cls_loss.append(loss_item['cls_loss'] * self.loss_weights_dict['cls_loss'])
            dis_loss.append(loss_item['dis_loss'] * self.loss_weights_dict['dis_loss'])
            iou_loss.append(loss_item['iou_loss'] * self.loss_weights_dict['iou_loss'])
        cls_loss = torch.stack(cls_loss).sum()
        dis_loss = torch.stack(dis_loss).sum()
        iou_loss = torch.stack(iou_loss).sum()
        return cls_loss, dis_loss, iou_loss


class HungarianMatcher(object):
    def __init__(self,
                 cls_weights=1.0,
                 distance_weights=5.0,
                 iou_weights=2.0):
        super(HungarianMatcher, self).__init__()
        self.cls_weights = cls_weights
        self.distance_weights = distance_weights
        self.iou_weights = iou_weights
        self.box_similarity = generalized_box_iou

    @torch.no_grad()
    def __call__(self, outputs, targets):
        """
        :param outputs: dict contains "pred_logits" and "pred_boxes"
                        "pred_logits" [bs,queries,num_cls]
                        "pred_boxes" [bs,queries,4]
        :param targets:list of dict witch contains "labels" and "boxes" len(targets) = bs
                        "labels" [gt_num,]
                        "boxes" [gt_num,4]
        :return:
        """

        match_idx = list()
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        if pred_logits.dtype == torch.float16:
            pred_logits = pred_logits.float()
        if pred_boxes.dtype == torch.float16:
            pred_boxes = pred_boxes.float()
        pred_logits = pred_logits.softmax(dim=-1)
        pred_boxes = pred_boxes.sigmoid()
        for idx, (pred_logits_item, pred_boxes_item, tgt_item) in enumerate(zip(pred_logits, pred_boxes, targets)):
            tgt_labels_item, tgt_boxes_item = tgt_item['labels'], tgt_item['boxes']
            if len(tgt_boxes_item) == 0:
                match_idx.append(([], []))
                continue
            cls_cost = -pred_logits_item[:, tgt_labels_item.long()]
            distance_cost = torch.cdist(pred_boxes_item, xyxy2xywh(tgt_boxes_item), p=1)
            iou_cost = -self.box_similarity(xywh2xyxy(pred_boxes_item), tgt_boxes_item)
            cost = self.cls_weights * cls_cost + self.distance_weights * distance_cost + self.iou_weights * iou_cost
            i, j = linear_sum_assignment(cost.cpu())
            match_idx.append((
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            ))
        return match_idx
