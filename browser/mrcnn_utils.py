import sys
sys.path.append('../models/Mask_RCNN')

import numpy as np

import mrcnn.model as modellib
from mrcnn.config import Config 
from utils import pad_image, unpad_image, merge_image_and_mask

HEIGHT, WIDTH = 520, 704
HEIGHT_TARGET, WIDTH_TARGET = 576, 704
MRCNN_WEIGHT = '../models/weight/mask_rcnn_l2reg-and-head_0010.h5'

def init_and_load_weight():

    class InferenceConfig(Config):
        NAME = 'l2reg-and-head'

        # Set batch size to 1
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # No. classes
        NUM_CLASSES = 4

        # Image dimensions
        IMAGE_RESIZE_MODE = "none"
        IMAGE_MIN_DIM = HEIGHT_TARGET
        IMAGE_MAX_DIM = WIDTH_TARGET
        IMAGE_SHAPE = [HEIGHT_TARGET, WIDTH_TARGET, 3]

        # Mini mask
        USE_MINI_MASK = False

    infer_config = InferenceConfig()

    mrcnn_model = modellib.MaskRCNN(mode="inference", config=infer_config, model_dir='./')
    mrcnn_model.load_weights(MRCNN_WEIGHT, by_name=True)

    return mrcnn_model

def prepare_input(image):
    """
    Preprocess input to predict
    """

    image = pad_image(image, 128, (HEIGHT_TARGET, WIDTH_TARGET), (HEIGHT, WIDTH))
    image = np.stack([image, image, image], axis=2)

    return image

def process_to_visualize_output(img, mask):
    img = unpad_image(img, (HEIGHT_TARGET, WIDTH_TARGET), (HEIGHT, WIDTH))
    mask = unpad_image(mask, (HEIGHT_TARGET, WIDTH_TARGET),
                      (HEIGHT, WIDTH))
    merged = merge_image_and_mask(img, mask)
    
    return merged

def count_num_ins_detect(ins_detect):
    ins_count = {}

    ins_dict = {1: 'astro', 2: 'cort', 3: 'shsy5y'}

    for ins in ins_detect:
        # checking whether it is in the dict or not
        if ins in ins_count:
            # incerementing the count by 1
            ins_count[ins] += 1
        else:
            # setting the count to 1
            ins_count[ins] = 1

    return ins_count, ins_dict 

def print_num_ins_detect(ins_count, ins_dict):
    s = ''
    for key, value in ins_count.items():
        s += '{} tế bào {}, '.format(value, ins_dict[key])
    return s[:-2]

