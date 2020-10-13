import math
import torch
import cv2 as cv
import numpy as np
from commons.augmentations import BoxInfo


class TensorList(object):
    def __init__(self, ):
        self.tensors = None
        self.labels = None
        self.boxes = None
        self.weights = None
        self.masks = None
        self.extras = dict()

    def set_tensors(self, img_list):
        tensors = list()
        for item in img_list:
            tensors.append(torch.from_numpy(np.ascontiguousarray(item.transpose(2, 0, 1))).float())
        self.tensors = torch.stack(tensors, dim=0)

    def set_mask(self, mask_list):
        masks = list()
        for item in mask_list:
            masks.append(torch.from_numpy(item).bool())
        self.masks = torch.stack(masks, dim=0)

    def set_labels(self, label_list):
        labels = list()
        for item in label_list:
            labels.append(torch.from_numpy(item).float())
        self.labels = labels

    def set_boxes(self, box_list):
        boxes = list()
        for item in box_list:
            boxes.append(torch.from_numpy(item).float())
        self.boxes = boxes

    def set_weights(self, weight_list):
        weights = list()
        for item in weight_list:
            weights.append(torch.from_numpy(item).float())
        self.weights = weights

    def set_extra(self, name, extra_list):
        extras = list()
        for item in extra_list:
            extras.append(torch.from_numpy(item).float())
        self.extras[name] = torch.stack(extras, dim=0)

    def to(self, device):
        self.tensors = self.tensors.to(device)
        self.masks = self.masks.to(device)
        if self.boxes is not None:
            self.boxes = [item.to(device) for item in self.boxes]
        if self.labels is not None:
            self.labels = [item.to(device) for item in self.labels]
        if self.weights is not None:
            self.weights = [item.to(device) for item in self.weights]


class BatchPadding(object):
    def __init__(self,
                 rgb_mean=(0.485, 0.456, 0.406),
                 rgb_std=(0.229, 0.224, 0.225),
                 center_padding=False,
                 size_divisible=32):
        super(BatchPadding, self).__init__()
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.center_padding = center_padding
        self.size_divisible = size_divisible

    def __call__(self, box_infos):
        hw_shapes = np.array([(box_info.img.shape[:2]) for box_info in box_infos])
        max_h, max_w = hw_shapes.max(axis=0)
        max_h = int(math.ceil(max_h / self.size_divisible) * self.size_divisible)
        max_w = int(math.ceil(max_w / self.size_divisible) * self.size_divisible)
        for box_info in box_infos:
            h, w = box_info.img.shape[:2]
            dw, dh = max_w - w, max_h - h
            if self.center_padding:
                dw /= 2
                dh /= 2
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            else:
                top, bottom = 0, dh
                left, right = 0, dw
            ret_img = cv.copyMakeBorder(box_info.img,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        cv.BORDER_CONSTANT,
                                        value=BoxInfo.EMPTY_INDEX)

            if box_info.box is not None and len(box_info.box):
                box_info.box[:, [0, 2]] = box_info.box[:, [0, 2]] + left
                box_info.box[:, [1, 3]] = box_info.box[:, [1, 3]] + top

            # box_info.img = ret_img
            # from datasets.coco import colors, coco_names
            # ret_img = box_info.draw_box(colors, coco_names)
            # import uuid
            # cv.imwrite("{:s}.jpg".format(str(uuid.uuid4()).replace('-', "")), ret_img)
            box_info.mask = ret_img[..., 0] != BoxInfo.EMPTY_INDEX
            zero_mask = ret_img == BoxInfo.EMPTY_INDEX
            ret_img = ((ret_img[..., ::-1] / 255.0) - np.array(self.rgb_mean)) / np.array(self.rgb_std)
            ret_img[zero_mask] = 0.
            box_info.img = ret_img
            box_info.extra = np.array([left, top, w, h])

        tensor_list = TensorList()
        tensor_list.set_tensors([item.img for item in box_infos])
        tensor_list.set_boxes([item.box for item in box_infos])
        tensor_list.set_labels([item.label for item in box_infos])
        tensor_list.set_weights([item.weights for item in box_infos])
        tensor_list.set_mask([item.mask for item in box_infos])
        tensor_list.set_extra('padding_shapes', [item.extra for item in box_infos])
        return tensor_list
