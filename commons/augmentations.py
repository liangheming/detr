import math
import random
import cv2 as cv
import numpy as np

cv.setNumThreads(0)


class BoxInfo(object):
    EMPTY_INDEX = -1

    def __init__(self, img, box, label, weights=None):
        """
        :param img: np.ndarray (color channel:bgr)
        :param box: np.ndarray [x1,y1,x2,y2]
        :param label: np.ndarray
        :param weights:
        """
        super(BoxInfo, self).__init__()
        self.img = img.astype(np.float32)
        self.box = box
        self.label = label
        self.weights = weights

    def valid_box_info(self):
        return self.img is not None

    def draw_box(self, colors, names):
        assert self.img is not None and self.box is not None and self.label is not None
        ret_img = self.img.copy()
        ret_img[ret_img == self.EMPTY_INDEX] = 0
        ret_img = ret_img.astype(np.uint8)
        if not len(self.box):
            return ret_img
        for label_idx, weight, (x1, y1, x2, y2) in zip(self.label, self.weights, self.box):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(ret_img, (x1, y1), (x2, y2), color=colors[int(label_idx)], thickness=2)
            cv.putText(ret_img, "{:s}".format(names[int(label_idx)]),
                       (x1, y1 + 5),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       colors[int(label_idx)], 2)
            cv.putText(ret_img, "{:.2f}".format(weight),
                       (x1, y1 + 15),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       colors[int(label_idx)], 2)
        return ret_img


class RandTransForm(object):
    def __init__(self, p=0.5):
        self.p = p

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        pass

    def __call__(self, box_info: BoxInfo) -> BoxInfo:
        assert box_info.valid_box_info(), "not a valid BoxInfo"
        aug_p = np.random.uniform()
        if aug_p <= self.p:
            box_info = self.aug(box_info)
        return box_info

    def reset(self, **settings):
        p = settings.get('p', None)
        if p is not None:
            self.p = p
        return self


class RandNoise(RandTransForm):
    def __init__(self, **kwargs):
        super(RandNoise, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img):
        mu = 0
        sigma = np.random.uniform(1, 15)
        img += np.random.normal(mu, sigma, img.shape)
        img = img.clip(0., 255.)
        return img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandBlur(RandTransForm):
    """
    随机进行模糊
    """

    def __init__(self, **kwargs):
        super(RandBlur, self).__init__(**kwargs)

    @staticmethod
    def gaussian_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    @staticmethod
    def median_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.medianBlur(img, kernel_size, 0)
        return img

    @staticmethod
    def blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.blur(img, (kernel_size, kernel_size))
        return img

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        aug_blur = np.random.choice([self.gaussian_blur, self.median_blur, self.blur])
        img = aug_blur(img)
        return img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class HSV(RandTransForm):
    """
    color jitter
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, **kwargs):
        super(HSV, self).__init__(**kwargs)
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        img = img.astype(np.uint8)
        hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
        ret_img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        ret_img = ret_img.astype(np.float32)
        return ret_img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class Identity(RandTransForm):
    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(Identity, self).__init__(**kwargs)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        return box_info


class ScaleMinMax(RandTransForm):
    def __init__(self, min_thresh=640, max_thresh=1024, rand_scale=False, **kwargs):
        kwargs['p'] = 1.0
        super(ScaleMinMax, self).__init__(**kwargs)
        assert min_thresh <= max_thresh
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.rand_scale = rand_scale

    def scale_img(self, img: np.ndarray):
        h, w = img.shape[:2]
        min_side, max_side = min(h, w), max(h, w)
        if not self.rand_scale:
            r = min(self.min_thresh / min_side, self.max_thresh / max_side)
        else:
            r_l, r_r = self.min_thresh / min_side, self.max_thresh / max_side
            r = np.random.uniform(r_l, r_r) if r_l < r_r else r_r
        if r != 1:
            img = cv.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv.INTER_LINEAR)
        return img, r

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        img, ratio = self.scale_img(box_info.img)
        box_info.img = img
        if box_info.box is not None and len(box_info.box):
            box_info.box = box_info.box * ratio
        return box_info


class ScaleMax(RandTransForm):
    def __init__(self, max_thresh=640,
                 pad_to_square=True,
                 minimum_rectangle=False,
                 scale_up=True, **kwargs):
        kwargs['p'] = 1.0
        super(ScaleMax, self).__init__(**kwargs)
        self.max_thresh = max_thresh
        self.pad_to_square = pad_to_square
        self.minimum_rectangle = minimum_rectangle
        self.scale_up = scale_up

    def make_border(self, img: np.ndarray, border_val):
        h, w = img.shape[:2]
        r = min(self.max_thresh / h, self.max_thresh / w)
        if not self.scale_up:
            r = min(r, 1.0)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if r != 1.0:
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        if not self.pad_to_square:
            return img, r, (0, 0)
        dw, dh = int(self.max_thresh - new_w), int(self.max_thresh - new_h)
        if self.minimum_rectangle:
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)
        dw /= 2
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=border_val)
        return img, r, (left, top)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        img, r, (left, top) = self.make_border(box_info.img, box_info.EMPTY_INDEX)
        box_info.img = img
        if box_info.box is not None and len(box_info.box):
            box_info.box = box_info.box * r
            box_info.box[:, [0, 2]] = box_info.box[:, [0, 2]] + left
            box_info.box[:, [1, 3]] = box_info.box[:, [1, 3]] + top
        return box_info

    def reset(self, **settings):
        super(ScaleMax, self).reset(**settings)
        max_thresh = settings.get('max_thresh', None)
        if max_thresh is not None:
            self.max_thresh = max_thresh
        return self


class LRFlip(RandTransForm):
    """
    左右翻转
    """

    def __init__(self, **kwargs):
        super(LRFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        img = np.fliplr(img)
        return img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        _, w = box_info.img.shape[:2]
        box_info.img = self.img_aug(box_info.img)
        if box_info.box is not None and len(box_info.box):
            box_info.box[:, [2, 0]] = w - box_info.box[:, [0, 2]]
        return box_info


class UDFlip(RandTransForm):
    """
    上下翻转
    """

    def __init__(self, **kwargs):
        super(UDFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        img = np.flipud(img)
        return img

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        h, _ = box_info.img.shape[:2]
        box_info.img = self.img_aug(box_info.img)
        if box_info.box is not None and len(box_info.box):
            box_info.box[:, [3, 1]] = h - box_info.box[:, [1, 3]]
        return box_info


class RandPerspective(RandTransForm):
    def __init__(self, target_size=(640, 640),
                 degree=(0, 0),
                 translate=0,
                 scale=(1.0, 1.0),
                 shear=0,
                 perspective=0.0, **kwargs):
        kwargs['p'] = 1.0
        super(RandPerspective, self).__init__(**kwargs)
        assert isinstance(target_size, tuple) or target_size is None
        assert isinstance(degree, tuple)
        assert isinstance(scale, tuple)
        self.target_size = target_size
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def reset(self, **settings):
        super(RandPerspective, self).reset(**settings)
        target_size = settings.get('target_size', None)
        degree = settings.get('degree', None)
        translate = settings.get('translate', None)
        scale = settings.get('scale', None)
        shear = settings.get('shear', None)
        perspective = settings.get('perspective', None)
        if target_size is not None:
            assert isinstance(target_size, tuple)
            self.target_size = target_size
        if degree is not None:
            assert isinstance(degree, tuple)
            self.degree = degree
        if translate is not None:
            self.translate = translate
        if scale is not None:
            assert isinstance(scale, tuple)
            self.scale = scale
        if shear is not None:
            self.shear = shear
        if perspective is not None:
            self.perspective = perspective
        return self

    def get_transform_matrix(self, img):
        if self.target_size is not None:
            width, height = self.target_size
        else:
            height, width = img.shape[:2]

        matrix_c = np.eye(3)
        matrix_c[0, 2] = -img.shape[1] / 2
        matrix_c[1, 2] = -img.shape[0] / 2

        matrix_p = np.eye(3)
        matrix_p[2, 0] = random.uniform(-self.perspective, self.perspective)
        matrix_p[2, 1] = random.uniform(-self.perspective, self.perspective)

        matrix_r = np.eye(3)
        angle = np.random.uniform(self.degree[0], self.degree[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        matrix_r[:2] = cv.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

        matrix_t = np.eye(3)
        matrix_t[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        matrix_t[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        matrix_s = np.eye(3)
        matrix_s[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        matrix_s[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        return matrix_t @ matrix_s @ matrix_r @ matrix_p @ matrix_c, width, height, scale

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        transform_matrix, width, height, scale = self.get_transform_matrix(box_info.img)
        if self.perspective:
            box_info.img = cv.warpPerspective(box_info.img,
                                              transform_matrix,
                                              dsize=(width, height),
                                              borderValue=box_info.EMPTY_INDEX)
        else:  # affine
            box_info.img = cv.warpAffine(box_info.img,
                                         transform_matrix[:2],
                                         dsize=(width, height),
                                         borderValue=box_info.EMPTY_INDEX)
        if box_info.box is None or len(box_info.box) == 0:
            return box_info
        n = len(box_info.box)
        if n:
            xy = np.ones((n * 4, 3))
            # x1,y1,x2,y2,x1,y2,x2,y1
            xy[:, :2] = box_info.box[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ transform_matrix.T)
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (box_info.box[:, 2] - box_info.box[:, 0]) * (box_info.box[:, 3] - box_info.box[:, 1])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 2) & (h > 2) & (area / (area0 * scale + 1e-16) > 0.2) & (ar < 20)
            box_info.box = xy[i]
            box_info.label = box_info.label[i]
            if box_info.weights is not None:
                box_info.weights = box_info.weights[i]
            return box_info


class Mosaic(RandTransForm):
    def __init__(self,
                 candidate_img_paths,
                 candidate_labels,
                 color_gitter=None,
                 target_size=640,
                 rand_center=True, **kwargs):
        """
        :param candidate_img_paths:list(img_paths)
        :param candidate_labels: list(label) (label_num,5) [label_idx,x1,y1,x2,y2]
        :param color_gitter:
        :param target_size:
        :param rand_center:
        :param kwargs:
        """
        kwargs['p'] = 1.0
        super(Mosaic, self).__init__(**kwargs)
        self.candidate_img_paths = candidate_img_paths
        self.candidate_labels = candidate_labels
        if color_gitter is None:
            color_gitter = Identity()
        self.color_gitter = color_gitter
        self.target_size = target_size
        self.rand_center = rand_center
        self.affine = RandPerspective(target_size=(target_size, target_size), translate=0.1, scale=(0.8, 1.0))
        self.scale_max = ScaleMax(max_thresh=target_size, pad_to_square=False)

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        mosaic_border = (-self.target_size // 2, -self.target_size // 2)
        if self.rand_center:
            yc, xc = [int(random.uniform(-x, 2 * self.target_size + x)) for x in mosaic_border]
        else:
            yc, xc = [self.target_size, self.target_size]
        indices = [random.randint(0, len(self.candidate_labels) - 1) for _ in range(3)]
        img4 = np.ones(shape=(self.target_size * 2, self.target_size * 2, 3)) * box_info.EMPTY_INDEX
        box_info4 = list()
        for i, index in enumerate([1] + indices):
            if i == 0:
                box_info_i = box_info
            else:
                img = cv.imread(self.candidate_img_paths[index])
                label_info = self.candidate_labels[index].copy()
                box_info_i = BoxInfo(img, label_info[:, 1:], label_info[:, 0], weights=np.ones_like(label_info[:, 0]))
            box_info_i = self.color_gitter(box_info_i)
            box_info_i = self.scale_max(box_info_i)
            h, w = box_info_i.img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.target_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.target_size * 2), min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = box_info_i.img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if box_info_i.box is not None and len(box_info_i.box):
                box_info_i.box[:, [0, 2]] = box_info_i.box[:, [0, 2]] + padw
                box_info_i.box[:, [1, 3]] = box_info_i.box[:, [1, 3]] + padh
                box_info4.append(box_info_i)
        box_info.img = img4
        if len(box_info4):
            box_4 = np.concatenate([item.box for item in box_info4], axis=0)
            np.clip(box_4, 0, 2 * self.target_size, out=box_4)
            label_4 = np.concatenate([item.label for item in box_info4], axis=0)
            weights_4 = np.concatenate([item.weights for item in box_info4], axis=0)
            box_info.box = box_4
            box_info.label = label_4
            box_info.weights = weights_4
        else:
            box_info.box = np.zeros(shape=(0, 4), dtype=np.float32)
            box_info.label = np.zeros(shape=(0,), dtype=np.float32)
            box_info.weights = np.zeros(shape=(0,), dtype=np.float32)

            return self.affine(box_info)
        valid_index = np.bitwise_and((box_info.box[:, 2] - box_info.box[:, 0]) > 2,
                                     (box_info.box[:, 3] - box_info.box[:, 1]) > 2)
        box_info.box = box_info.box[valid_index, :]
        box_info.label = box_info.label[valid_index]
        box_info.weights = box_info.weights[valid_index]
        return self.affine(box_info)


class MosaicWrapper(Mosaic):
    def __init__(self, min_thresh=640, max_thresh=1024, **kwargs):
        super(MosaicWrapper, self).__init__(**kwargs)
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        rand_size = int(np.random.uniform(self.min_thresh, self.max_thresh))
        self.target_size = rand_size
        self.affine.reset(target_size=(rand_size, rand_size))
        self.scale_max.reset(max_thresh=rand_size)
        return super(MosaicWrapper, self).aug(box_info)


class MixUp(RandTransForm):
    def __init__(self, candidate_img_paths,
                 candidate_labels,
                 color_gitter=None,
                 mix_ratio=0.5, **kwargs):
        kwargs['p'] = 1.0
        super(MixUp, self).__init__(**kwargs)
        self.candidate_img_paths = candidate_img_paths
        self.candidate_labels = candidate_labels
        if color_gitter is None:
            color_gitter = Identity()
        self.color_gitter = color_gitter
        self.mix_ratio = mix_ratio

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        index = random.randint(0, len(self.candidate_labels) - 1)
        append_img = cv.imread(self.candidate_img_paths[index])
        append_labels = self.candidate_labels[index].copy()
        append_box_info = BoxInfo(append_img,
                                  append_labels[:, 1:],
                                  append_labels[:, 0],
                                  weights=np.ones_like(append_labels[:, 0]))
        box_info = self.color_gitter(box_info)
        append_box_info = self.color_gitter(append_box_info)
        h1, w1 = box_info.img.shape[:2]
        h2, w2 = append_box_info.img.shape[:2]
        temp_img = np.ones(shape=(max(h1, h2), max(w1, w2), 3), dtype=np.float32) * box_info.EMPTY_INDEX
        inner_w, inner_h = min(w1, w2), min(h1, h2)
        temp_img[0:h1, 0:w1, :] = self.mix_ratio * box_info.img
        temp_img[0:h2, 0:w2, :] = (1 - self.mix_ratio) * append_box_info.img
        temp_img[0:inner_h, 0:inner_w, :] = self.mix_ratio * box_info.img[0:inner_h, 0:inner_w, :] + (
                1 - self.mix_ratio) * append_box_info.img[0:inner_h, 0:inner_w, :]
        box_info.img = temp_img
        if box_info.box is not None and len(box_info.box):
            box_info.box = np.concatenate([box_info.box, append_box_info.box], axis=0)
            box_info.label = np.concatenate([box_info.label, append_box_info.label], axis=0)
            box_info.weights = np.concatenate([box_info.weights * self.mix_ratio,
                                               append_box_info.weights * (1 - self.mix_ratio)], axis=0)
        return box_info


class MixUpWrapper(MixUp):
    def __init__(self, beta=(8, 8), **kwargs):
        super(MixUpWrapper, self).__init__(**kwargs)
        self.beta = beta

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        mix_ratio = np.random.beta(self.beta[0], self.beta[1])
        self.mix_ratio = mix_ratio
        return super(MixUpWrapper, self).aug(box_info)


class OneOf(RandTransForm):
    def __init__(self, transforms, **kwargs):
        kwargs['p'] = 1.0
        super(OneOf, self).__init__(**kwargs)
        if isinstance(transforms[0], RandTransForm):
            prob = float(1 / len(transforms))
            transforms = [(prob, transform) for transform in transforms]
        probs, transforms = zip(*transforms)
        probs, transforms = list(probs), list(transforms)
        self.probs = probs
        self.transforms = transforms

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        index = np.random.choice(a=range(len(self.probs)), p=self.probs)
        box_info = self.transforms[index](box_info)
        return box_info


class Compose(RandTransForm):
    def __init__(self, transforms, **kwargs):
        kwargs['p'] = 1.0
        super(Compose, self).__init__(**kwargs)
        self.transforms = transforms

    def aug(self, box_info: BoxInfo) -> BoxInfo:
        for transform in self.transforms:
            box_info = transform(box_info)
        return box_info
