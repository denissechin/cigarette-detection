import random
from typing import List

import cv2
import numpy as np


def rotate_cigarette(image: np.ndarray, rotate_limit: int, attachment_point: List[int]):
    """
    Rotates image around center NOT keeping original size

    Args:
        image: image to rotate
        rotate_limit: limit to pick rotation angle from, (-rotate_limit, rotate_limit)
        attachment_point: scaled point of cigarette attachment

    Returns tuple of:
        rotated_image: resulting rotated image
        attachment_bias: rotated attachment point

    """
    rotation_angle = random.randint(-rotate_limit, rotate_limit)

    height, width = image.shape[:2]
    rotate_point = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(rotate_point, rotation_angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - rotate_point[0]
    rotation_mat[1, 2] += bound_h / 2 - rotate_point[1]

    # Also rotate attachment point that is by default in center of cigarette image
    attachment_bias = cv2.transform(np.array([[attachment_point]]), rotation_mat).squeeze()
    rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))

    return rotated_image, attachment_bias


def overlay_image_alpha(image: np.ndarray, image_overlay: np.ndarray, point: List[int]):
    """

    Args:
        image: RGB image to be overlayed by image_overlay
        image_overlay: RGBA image to overlay on image
        point: (xmin, ymin) point where image_overlay is attached

    Returns:
        img: overlayed image
    """

    img = image.copy()
    x, y = point
    alpha_mask = image_overlay[:, :, 3] / 255
    image_overlay_rgb = image_overlay[:, :, :3]
    y1, y2 = max(0, y), min(img.shape[0], y + image_overlay_rgb.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + image_overlay_rgb.shape[1])
    y1o, y2o = max(0, -y), min(image_overlay_rgb.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(image_overlay_rgb.shape[1], img.shape[1] - x)

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = image_overlay_rgb[y1o:y2o, x1o:x2o]
    if img_crop.shape != img_overlay_crop.shape:
        return None
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    return img


def xyxy2xywh(box):
    """
    xmin, ymin, xmax, ymax -> xmin, ymin, width, height (COCO)
    """
    a = box[:2]
    b = box[2:]
    transformed_box = np.concatenate([a, b - a], 0).astype(int).tolist()
    return transformed_box
