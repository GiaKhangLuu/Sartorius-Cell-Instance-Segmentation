import numpy as np
import cv2

def pad_image(image, constant_values, SHAPE_TARGET, SHAPE):
    """
    Func. to pad images and masks
    """

    HEIGHT_TARGET, WIDTH_TARGET = SHAPE_TARGET
    HEIGHT, WIDTH = SHAPE

    pad_h = (HEIGHT_TARGET - HEIGHT) // 2
    pad_w = (WIDTH_TARGET - WIDTH) // 2
    
    if len(image.shape) == 3:
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), constant_values=constant_values)
    else:
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=constant_values)

def unpad_image(image, SHAPE_TARGET, SHAPE):
    """
    Removes the padding from an image
    """

    HEIGHT_TARGET, WIDTH_TARGET = SHAPE_TARGET
    HEIGHT, WIDTH = SHAPE
    offset_h = (HEIGHT_TARGET - HEIGHT) // 2
    offset_w = (WIDTH_TARGET - WIDTH) // 2
    
    return image[offset_h:offset_h+HEIGHT, offset_w:offset_w+WIDTH]

def merge_image_and_mask(image, mask):
    """
    Stack image input and predicted mask to visualize
    """

    image = image[..., 0] if image.ndim == 3 else image
    mask = np.sum(mask, -1) if mask.ndim == 3 else mask

    merged = cv2.addWeighted(image, 0.75,
                             np.clip(mask.astype(image.dtype), 0, 1)*255, 
                             0.25, 0.0,)

    return merged
