import cv2
import numpy as np
from enum import Enum

def compute_bounding_box(results):
    return cv2.boundingRect(results.segmentation_mask.astype(np.uint8))

def add_occlusion(image, bounding_box, percent: float=0.2):
    image = image.copy()
    x, y, w, h = bounding_box
    new_w = int(w * percent)
    new_h = int(h * percent)
    occlusion = np.zeros((h, w, 3), dtype=np.uint8)
    occlusion.fill(255)
    occlusion = cv2.resize(occlusion, (new_w, new_h))
    
    x_loc = np.random.randint(x, x+w-new_w)
    y_loc = np.random.randint(y, y+h-new_h)

    image[y_loc:y_loc+new_h, x_loc:x_loc+new_w] = occlusion

    return image