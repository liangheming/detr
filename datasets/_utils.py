import math
import numpy as np
from commons.augmentations import BoxInfo
import cv2 as cv


class TensorList(object):
    def __init__(self, ):
        self.tensors = None
        self.labels = None
        self.boxes = None
        self.weights = None
        self.shapes = None

    @staticmethod
    def build_tensor_list(img_list, label_list, box_list, weight_list):
        pass

    def set_tensors(self, img_list):
        pass

    def set_labels(self, label_list):
        pass

    def set_boxes(self, box_list):
        pass

    def set_weights(self, weight_list):
        pass


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
            box_info.img = ret_img
            if box_info.box is not None and len(box_info.box):
                box_info.box[:, [0, 2]] = box_info.box[:, [0, 2]] + left
                box_info.box[:, [1, 3]] = box_info.box[:, [1, 3]] + top
            from datasets.coco import colors, coco_names
            ret_img = box_info.draw_box(colors, coco_names)
            import uuid
            cv.imwrite("{:s}.jpg".format(str(uuid.uuid4()).replace('-', "")), ret_img)

