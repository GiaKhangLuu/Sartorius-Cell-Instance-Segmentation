import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
import tifffile
import glob
import seaborn as sns
import cv2
from pycocotools import mask as coco_mask
import imageio
from tqdm import tqdm


class LiveCell_Dataset:
    
    def __init__(self, image_folder, train_annotation_path, val_annotation_path,
            shape_target, shape):
        """
        Args:
            shape_target: [height_target, width_target]
            shape: [height, width]
        """
        self.shape_target, self.shape = shape_target, shape
        self.image_folder = image_folder
        self.train_annotation_path = train_annotation_path
        self.val_annotation_path = val_annotation_path
        self._create_dataframe()
    
    def _create_dataframe(self):
        """
        Create DataFrame for later use
        """

        train_annot_file = open(self.train_annotation_path)
        val_annot_file = open(self.val_annotation_path)

        train_annotation = json.load(train_annot_file)
        val_annotation = json.load(val_annot_file)
        
        self.train_annot_df = pd.DataFrame(train_annotation['annotations']).transpose()
        self.train_image_df = pd.DataFrame(train_annotation['images'])

        self.val_annot_df = pd.DataFrame(val_annotation['annotations']).transpose()
        self.val_image_df = pd.DataFrame(val_annotation['images'])

        assert self.train_annot_df['image_id'].nunique() == len(self.train_image_df)
        assert self.val_annot_df['image_id'].nunique() == len(self.val_image_df)

        #return train_annot_df, train_image_df, val_annot_df, val_image_df

    def rle_encode(self, img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''

        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]

        return ' '.join(str(x) for x in runs)

    def convert_polygon_to_rle(self):
        """
        Convert polygon annotation to rle
        """

        tqdm.pandas()
        def polygon_to_rle(annot_polygon):
            rle = coco_mask.frPyObjects(annot_polygon, *self.shape)
            mask = coco_mask.decode(rle)
            rle = self.rle_encode(mask)

            return rle
        
        self.train_annot_df['segmentation'] = self.train_annot_df['segmentation'].progress_apply(polygon_to_rle)
        self.val_annot_df['segmentation'] = self.val_annot_df['segmentation'].progress_apply(polygon_to_rle)
       
    def pad_image(self, image, constant_values):
        """
        Padding image to expected shape
        """

        HEIGHT_TARGET, WIDTH_TARGET = self.shape_target
        HEIGHT, WIDTH = self.shape
        pad_h = (HEIGHT_TARGET - HEIGHT) // 2
        pad_w = (WIDTH_TARGET - WIDTH) // 2
    
        if len(image.shape) == 3:
            return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), constant_values=constant_values)
        else:
            return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=constant_values)

    def save_image_to_working_dir(self, img_group='train', pad_img=False):
        """
        Save images to working directory
        """
        
        assert img_group == 'train' or img_group == 'val'

        df = self.train_image_df if img_group == 'train' else self.val_image_df

        for row in tqdm(df.itertuples(index=False)):
            img_path = os.path.join(self.image_folder, row.file_name)
    
            # Read file
            img = tifffile.imread(img_path)

            img = self.pad_image(img, 128) if pad_img else img
    
            # Save file
            img_file = row.original_filename
            des_path = './{}/{}'.format(img_group, img_file)
            imageio.imsave(des_path, img) 

    def create_livecell_dataframe(self, img_group='train'):
        """
        Create LiveCell dataframe contains 3 columns: id, annotation, cell_type.
        Note: segmentation column in train_annot_df have to be converted to rle 
        before creating livecell dataframe
        """
        
        assert img_group == 'train' or img_group == 'val'

        img_df = self.train_image_df if img_group == 'train' else self.val_image_df
        annot_df = self.train_annot_df if img_group == 'train' else self.val_annot_df

        # Remove '.png' from original_filename
        image_dict = dict([(k, v[:-4]) for k, v in img_df[['id', 'original_filename']].itertuples(index=False)])

        img_file, annotation = [], []

        for row in tqdm(annot_df.itertuples(index=False)):
            annotation.append(row.segmentation)
            img_file.append(image_dict[row.image_id])        

        cell_type = ['shsy5y'] * len(img_file)
        
        livecell_df = pd.DataFrame({'id': img_file, 'annotation': annotation, 'cell_type': cell_type})

        return livecell_df





