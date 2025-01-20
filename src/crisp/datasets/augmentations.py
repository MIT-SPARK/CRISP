import numpy as np
import PIL
import torch
import random
from PIL import ImageEnhance, ImageFilter
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from copy import deepcopy
import torchvision


def invert_T(T):
    R = T[..., :3, :3]
    t = T[..., :3, [-1]]
    R_inv = R.transpose(-2, -1)
    t_inv = -R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, [-1]] = t_inv
    return T_inv


def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4,)
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.tensor(orig_size, dtype=torch.float)
    crop_resize = torch.tensor(crop_resize, dtype=torch.float)

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K


def crop_to_aspect_ratio(images, box, masks=None, K=None):
    assert images.dim() == 4
    bsz, _, h, w = images.shape
    assert box.dim() == 1
    assert box.shape[0] == 4
    w_output, h_output = box[[2, 3]] - box[[0, 1]]
    boxes = torch.cat(
        (torch.arange(bsz).unsqueeze(1).to(box.device).float(), box.unsqueeze(0).repeat(bsz, 1).float()), dim=1
    ).to(images.device)
    images = torchvision.ops.roi_pool(images, boxes, output_size=(h_output, w_output))
    if masks is not None:
        assert masks.dim() == 4
        masks = torchvision.ops.roi_pool(masks, boxes, output_size=(h_output, w_output))
    if K is not None:
        assert K.dim() == 3
        assert K.shape[0] == bsz
        K = get_K_crop_resize(K, boxes[:, 1:], orig_size=(h, w), crop_resize=(h_output, w_output))
    return images, masks, K


def make_detections_from_segmentation(masks):
    detections = []
    if masks.dim() == 4:
        assert masks.shape[0] == 1
        masks = masks.squeeze(0)

    for mask_n in masks:
        dets_n = dict()
        for uniq in torch.unique(mask_n, sorted=True):
            ids = np.where((mask_n == uniq).cpu().numpy())
            x1, y1, x2, y2 = np.min(ids[1]), np.min(ids[0]), np.max(ids[1]), np.max(ids[0])
            dets_n[int(uniq.item())] = torch.tensor([x1, y1, x2, y2]).to(mask_n.device)
        detections.append(dets_n)
    return detections


def make_masks_from_det(detections, h, w):
    n_ids = len(detections)
    detections = torch.as_tensor(detections)
    masks = torch.zeros((n_ids, h, w)).byte()
    for mask_n, det_n in zip(masks, detections):
        x1, y1, x2, y2 = det_n.cpu().int().tolist()
        mask_n[y1:y2, x1:x2] = True
    return masks


def to_pil(im):
    if isinstance(im, PIL.Image.Image):
        return im
    elif isinstance(im, torch.Tensor):
        return PIL.Image.fromarray(np.asarray(im))
    elif isinstance(im, np.ndarray):
        return PIL.Image.fromarray(im)
    else:
        raise ValueError("Type not supported", type(im))


def to_torch_uint8(im):
    if isinstance(im, PIL.Image.Image):
        im = torch.as_tensor(np.asarray(im).astype(np.uint8))
    elif isinstance(im, torch.Tensor):
        assert im.dtype == torch.uint8
    elif isinstance(im, np.ndarray):
        assert im.dtype == np.uint8
        im = torch.as_tensor(im)
    else:
        raise ValueError("Type not supported", type(im))
    if im.dim() == 3:
        assert im.shape[-1] in {1, 3}
    return im


class PillowBlur:
    def __init__(self, p=0.4, factor_interval=(1, 3)):
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im, mask, obs):
        im = to_pil(im)
        k = random.randint(*self.factor_interval)
        im = im.filter(ImageFilter.GaussianBlur(k))
        return im, mask, obs


class PillowRGBAugmentation:
    def __init__(self, pillow_fn, p, factor_interval):
        self._pillow_fn = pillow_fn
        self.p = p
        self.factor_interval = factor_interval

    def __call__(self, im, mask, obs):
        im = to_pil(im)
        if random.random() <= self.p:
            im = self._pillow_fn(im).enhance(factor=random.uniform(*self.factor_interval))
        return im, mask, obs


class PillowSharpness(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 50.0)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness, p=p, factor_interval=factor_interval)


class PillowContrast(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.2, 50.0)):
        super().__init__(pillow_fn=ImageEnhance.Contrast, p=p, factor_interval=factor_interval)


class PillowBrightness(PillowRGBAugmentation):
    def __init__(self, p=0.5, factor_interval=(0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness, p=p, factor_interval=factor_interval)


class PillowColor(PillowRGBAugmentation):
    def __init__(self, p=0.3, factor_interval=(0.0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color, p=p, factor_interval=factor_interval)


class GrayScale(PillowRGBAugmentation):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, im, mask, obs):
        im = to_pil(im)
        if random.random() <= self.p:
            im = to_torch_uint8(im).float()
            gray = 0.2989 * im[..., 0] + 0.5870 * im[..., 1] + 0.1140 * im[..., 2]
            gray = gray.to(torch.uint8)
            im = gray.unsqueeze(-1).repeat(1, 1, 3)
        return im, mask, obs


class BackgroundAugmentation:
    def __init__(self, image_dataset, p):
        self.image_dataset = image_dataset
        self.p = p

    def get_bg_image(self, idx):
        return self.image_dataset[idx]

    def __call__(self, im, mask, obs):
        if random.random() <= self.p:
            im = to_torch_uint8(im)
            mask = to_torch_uint8(mask)
            h, w, c = im.shape
            im_bg = self.get_bg_image(random.randint(0, len(self.image_dataset) - 1))
            im_bg = to_pil(im_bg)
            im_bg = torch.as_tensor(np.asarray(im_bg.resize((w, h))))
            mask_bg = mask == 0
            im[mask_bg] = im_bg[mask_bg]
        return im, mask, obs


class VOCBackgroundAugmentation(BackgroundAugmentation):
    def __init__(self, voc_root, p=0.3):
        image_dataset = ImageFolder(voc_root)
        super().__init__(image_dataset=image_dataset, p=p)

    def get_bg_image(self, idx):
        return self.image_dataset[idx][0]


class CropResizeToAspectAugmentation:
    def __init__(self, resize=(640, 480)):
        self.resize = (min(resize), max(resize))
        self.aspect = max(resize) / min(resize)

    def __call__(self, im, mask, obs):
        im = to_torch_uint8(im)
        mask = to_torch_uint8(mask)
        obs["orig_camera"] = deepcopy(obs["camera"])
        assert im.shape[-1] == 3
        h, w = im.shape[:2]
        if (h, w) == self.resize:
            obs["orig_camera"]["crop_resize_bbox"] = (0, 0, w - 1, h - 1)
            return im, mask, obs

        images = (torch.as_tensor(im).float() / 255).unsqueeze(0).permute(0, 3, 1, 2)
        masks = torch.as_tensor(mask).unsqueeze(0).unsqueeze(0).float()
        K = torch.tensor(obs["camera"]["K"]).unsqueeze(0)

        # Match the width on input image with an image of target aspect ratio.
        if not np.isclose(w / h, self.aspect):
            x0, y0 = images.shape[-1] / 2, images.shape[-2] / 2
            w = images.shape[-1]
            r = self.aspect
            h = w * 1 / r
            box_size = (h, w)
            h, w = min(box_size), max(box_size)
            x1, y1, x2, y2 = x0 - w / 2, y0 - h / 2, x0 + w / 2, y0 + h / 2
            box = torch.tensor([x1, y1, x2, y2])
            images, masks, K = crop_to_aspect_ratio(images, box, masks=masks, K=K)

        # Resize to target size
        x0, y0 = images.shape[-1] / 2, images.shape[-2] / 2
        h_input, w_input = images.shape[-2], images.shape[-1]
        h_output, w_output = min(self.resize), max(self.resize)
        box_size = (h_input, w_input)
        h, w = min(box_size), max(box_size)
        x1, y1, x2, y2 = x0 - w / 2, y0 - h / 2, x0 + w / 2, y0 + h / 2
        box = torch.tensor([x1, y1, x2, y2])
        images = F.interpolate(images, size=(h_output, w_output), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks, size=(h_output, w_output), mode="nearest")
        obs["orig_camera"]["crop_resize_bbox"] = tuple(box.tolist())
        K = get_K_crop_resize(K, box.unsqueeze(0), orig_size=(h_input, w_input), crop_resize=(h_output, w_output))

        # Update the bounding box annotations
        dets_gt = make_detections_from_segmentation(masks)[0]
        for n, obj in enumerate(obs["objects"]):
            if "bbox" in obj:
                assert "id_in_segm" in obj
                obj["bbox"] = dets_gt[obj["id_in_segm"]]

        im = (images[0].permute(1, 2, 0) * 255).to(torch.uint8)
        mask = masks[0, 0].to(torch.uint8)
        obs["camera"]["K"] = K.squeeze(0).numpy()
        obs["camera"]["resolution"] = (w_output, h_output)
        return im, mask, obs


class CenterCrop:
    def __init__(self, resize=(640, 480)):
        self.resize = (min(resize), max(resize))
        self.aspect = max(resize) / min(resize)

    def __call__(self, im, mask, obs):
        im = to_torch_uint8(im)
        mask = to_torch_uint8(mask)
        obs["orig_camera"] = deepcopy(obs["camera"])
        assert im.shape[-1] == 3
        h, w = im.shape[:2]
        if (h, w) == self.resize:
            obs["orig_camera"]["crop_resize_bbox"] = (0, 0, w - 1, h - 1)
            return im, mask, obs

        images = (torch.as_tensor(im).float() / 255).unsqueeze(0).permute(0, 3, 1, 2)
        masks = torch.as_tensor(mask).unsqueeze(0).unsqueeze(0).float()
        K = torch.tensor(obs["camera"]["K"]).unsqueeze(0)

        # Match the width on input image with an image of target aspect ratio.
        if not np.isclose(w / h, self.aspect):
            x0, y0 = images.shape[-1] / 2, images.shape[-2] / 2
            box_size = self.resize
            h, w = min(box_size), max(box_size)
            x1, y1, x2, y2 = x0 - w / 2, y0 - h / 2, x0 + w / 2, y0 + h / 2
            box = torch.tensor([x1, y1, x2, y2])
            images, masks, K = crop_to_aspect_ratio(images, box, masks=masks, K=K)

        # Resize to target size
        x0, y0 = images.shape[-1] / 2, images.shape[-2] / 2
        h_input, w_input = images.shape[-2], images.shape[-1]
        h_output, w_output = min(self.resize), max(self.resize)
        box_size = (h_input, w_input)
        h, w = min(box_size), max(box_size)
        x1, y1, x2, y2 = x0 - w / 2, y0 - h / 2, x0 + w / 2, y0 + h / 2
        box = torch.tensor([x1, y1, x2, y2])
        images = F.interpolate(images, size=(h_output, w_output), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks, size=(h_output, w_output), mode="nearest")
        obs["orig_camera"]["crop_resize_bbox"] = tuple(box.tolist())
        K = get_K_crop_resize(K, box.unsqueeze(0), orig_size=(h_input, w_input), crop_resize=(h_output, w_output))

        # Update the bounding box annotations
        dets_gt = make_detections_from_segmentation(masks)[0]
        for n, obj in enumerate(obs["objects"]):
            if "bbox" in obj:
                assert "id_in_segm" in obj
                obj["bbox"] = dets_gt[obj["id_in_segm"]]

        im = (images[0].permute(1, 2, 0) * 255).to(torch.uint8)
        mask = masks[0, 0].to(torch.uint8)
        obs["camera"]["K"] = K.squeeze(0).numpy()
        obs["camera"]["resolution"] = (w_output, h_output)
        return im, mask, obs
