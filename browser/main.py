######################################
#       IMPORT 
######################################

import numpy as np
import pandas as pd 
import streamlit as st
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

from utils import unpad_image, merge_image_and_mask, pad_image
import  mrcnn_utils 
import cellpose_utils

import sys
sys.path.append('../models/Mask_RCNN')
from mrcnn.visualize import display_instances, apply_mask, random_colors

sys.path.append('../models/cellpose')
from cellpose.plot import mask_overlay

######################################
#       DEFINE CONSTANCE 
######################################

HEIGHT, WIDTH = 520, 704
HEIGHT_TARGET, WIDTH_TARGET = 576, 704
IMG_DIR = '../sartorius-cell-instance-segmentation/train'

sartorius_df = pd.read_csv('../sartorius-cell-instance-segmentation/train.csv')

def rle_decode_by_image_id(image_id, shape=(520, 704)):
    SHAPE=shape
    rows = sartorius_df.loc[sartorius_df['id'] == image_id]

    # Image shape
    mask = np.full(shape=[len(rows), np.prod(SHAPE)], fill_value=0, dtype=np.uint8)

    for idx, (_, row) in enumerate(rows.iterrows()):
        s = row['annotation'].split()
        starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[idx, lo:hi] = True

    mask = mask.reshape([len(rows), *SHAPE])
    mask = np.moveaxis(mask, 0, 2)

    return mask

######################################
#   BROWSER  
######################################

st.set_page_config(layout='wide')

st.title('Phát hiện và khoanh vùng tế bào')

# Selected box
option = st.selectbox('Chọn mô hình dự đoán', ('Mask RCNN', 'Cellpose', 'Cả hai mô hình'))
st.write('Bạn chọn: ', option)

uploaded_file = st.file_uploader("Chọn ảnh")

# Button
if (option == 'Mask RCNN') and (uploaded_file is not None):

    cl_1, cr_1 = st.columns(2)
    
    # Load image input
    file_name = uploaded_file.name
    file_path = os.path.join(IMG_DIR, file_name)
    img = imageio.imread(file_path)
    
    # Load GROUND TRUTH
    num_of_ins_in_gt = str(len(sartorius_df[sartorius_df['id'] == file_name[:-4]]))
    mask_gt = rle_decode_by_image_id(file_name[:-4])
    colors = random_colors(mask_gt.shape[-1])
    gt = np.stack([img, img, img], axis=2)
    for i in range(len(colors)):
        m = mask_gt[..., i]
        color = colors[i]
        gt = apply_mask(gt, m, color, alpha=0.4)
    
    figsize = (10, 10)
    with cl_1:
        st.subheader('Input')
        st.write('Số lượng tế bào trong ảnh: **{}**'.format(num_of_ins_in_gt))
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        st.pyplot(fig)
    cl_2, cr_2 = st.columns(2)
    if st.button('Khoành vùng tế bào'):
        detect = None
        colors_display = None
        with cr_1:
            st.subheader('Output từ Mask RCNN')
            with st.spinner('Đang khởi tạo mô hình Mask RCNN'):
                mrcnn = mrcnn_utils.init_and_load_weight()
                mrcnn_input = mrcnn_utils.prepare_input(img)
            #st.success('Hoàn tất khởi tạo Mask RCNN')
        
            with st.spinner('Đang khoanh vùng bởi Mask RCNN'):
                detect = result = mrcnn.detect([mrcnn_input])[0]
            #st.success('Hoàn tất khoanh vùng')
            #merged = mrcnn_utils.process_to_visualize_output(mrcnn_input, result['masks'])
            #st.image(merged)
            
            num_ins_detect = str(result['masks'].shape[-1])
            st.write('Số lượng tế bào được tìm thấy bởi Mask RCNN: **{}**'.format(num_ins_detect))
        
            fig, ax = plt.subplots(figsize=figsize)
            mask = unpad_image(result['masks'], (HEIGHT_TARGET, WIDTH_TARGET),
                     (HEIGHT, WIDTH))
            input_display = unpad_image(mrcnn_input, (HEIGHT_TARGET, WIDTH_TARGET),
                    (HEIGHT, WIDTH))
            colors_display = colors = random_colors(mask.shape[-1])
            display_instances(input_display, result['rois'], mask, 
                    result['class_ids'], ['bg', 'astro', 'cort', 'shsy5y'], ax=ax,
                    alpha=0.2, remove_pad=True, colors=colors)
            st.pyplot(fig)
    
        st.header('So sánh Ground Truth và Output')
        with cl_2:
            st.write('Số lượng tế bào trong ảnh: **{}**'.format(num_of_ins_in_gt))
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(gt)
            plt.axis('off')
            st.pyplot(fig)
        with cr_2:
            st.write('Số lượng tế bào được tìm thấy bởi Mask RCNN: **{}**'.
                    format(detect['masks'].shape[-1]))
            fig, ax = plt.subplots(figsize=figsize)
            mask = unpad_image(detect['masks'], (HEIGHT_TARGET, WIDTH_TARGET),
                     (HEIGHT, WIDTH))
            input_display = unpad_image(mrcnn_input, (HEIGHT_TARGET, WIDTH_TARGET),
                    (HEIGHT, WIDTH))
            display_instances(input_display, detect['rois'], mask, 
                    result['class_ids'], ['bg', 'astro', 'cort', 'shsy5y'], ax=ax,
                    alpha=0.2, remove_pad=True, colors=colors_display)
            st.pyplot(fig)

if (option == 'Cellpose') and (uploaded_file is not None):

    cl_1, cr_1 = st.columns(2)
    
    # Load image input
    file_name = uploaded_file.name
    file_path = os.path.join(IMG_DIR, file_name)
    img = imageio.imread(file_path)
    
    # Load GROUND TRUTH
    num_of_ins_in_gt = str(len(sartorius_df[sartorius_df['id'] == file_name[:-4]]))
    mask_gt = rle_decode_by_image_id(file_name[:-4])
    colors = random_colors(mask_gt.shape[-1])
    gt = np.stack([img, img, img], axis=2)
    for i in range(len(colors)):
        m = mask_gt[..., i]
        color = colors[i]
        gt = apply_mask(gt, m, color, alpha=0.4)
    
    figsize = (10, 10)
    with cl_1:
        st.subheader('Input')
        st.write('Số lượng tế bào trong ảnh: **{}**'.format(num_of_ins_in_gt))
        st.image(img)
 
    cl_2, cr_2 = st.columns(2)
    if st.button('Khoành vùng tế bào'):
        detect = None
        with cr_1:
            st.subheader('Output từ Cellpose')
            with st.spinner('Khởi tạo Cellpose và Size Model'):
                cp_model, sz_model = cellpose_utils.init_models()
            #st.success('Hoàn tất khởi tạo Cellpose và Size Model')
            
            with st.spinner('Đang khoanh vùng bởi Cellpose và Size Model'):
                pred_diam, _ = sz_model.eval(img, channels=[0, 0])
                mask, flow, _ = cp_model.eval(img, channels=[0, 0],
                diameter=pred_diam, augment=True)
                detect = mask
            #st.success('Hoàn tất khoanh vùng')
            #merged = merge_image_and_mask(img, mask)
            #st.image(merged, caption='Output từ Cellpose')
            num_ins_detect = str(mask.max())
            st.write('Số lượng tế bào được tìm thấy bởi Cellpose: **{}**'.format(num_ins_detect))

            overlay = mask_overlay(img, mask)
            st.image(overlay)

        st.header('So sánh Ground Truth và Output')
        with cl_2:
            st.write('Số lượng tế bào trong ảnh: **{}**'.format(num_of_ins_in_gt))
            st.image(gt)
        with cr_2:
            st.write('Số lượng tế bào được tìm thấy bởi Cellpose: **{}**'.
                    format(detect.max()))
            overlay = mask_overlay(img, detect)
            st.image(overlay)

elif (option == 'Cả hai mô hình')and (uploaded_file is not None):
    # Load image input
    file_name = uploaded_file.name
    file_path = os.path.join(IMG_DIR, file_name)
    img = imageio.imread(file_path)
    st.subheader('Input')
    
    # Load GROUND TRUTH
    num_of_ins_in_gt = str(len(sartorius_df[sartorius_df['id'] == file_name[:-4]]))
    mask_gt = rle_decode_by_image_id(file_name[:-4])
    colors = random_colors(mask_gt.shape[-1])
    gt = np.stack([img, img, img], axis=2)
    for i in range(len(colors)):
        m = mask_gt[..., i]
        color = colors[i]
        gt = apply_mask(gt, m, color, alpha=0.4)

    st.write('Số lượng tế bào trong ảnh: **{}**'.format(num_of_ins_in_gt))
    st.image(img)

    st.header('So sánh Ground Truth và Output')
    col_gt, col_mrcnn, col_cp = st.columns(3) 

    if st.button('Khoành vùng tế bào'):

        with col_gt:
            st.subheader('Ảnh Ground Truth')
            st.write('Số lượng tế bào trong ảnh: **{}**'.format(num_of_ins_in_gt))
            st.image(gt)

        with col_mrcnn:
            st.subheader('Output từ Mask RCNN')
            with st.spinner('Khởi tạo mô hình Mask RCNN'):
                mrcnn = mrcnn_utils.init_and_load_weight()
                mrcnn_input = mrcnn_utils.prepare_input(img)
            #st.success('Hoàn tất khởi tạo Mask RCNN')
    
            with st.spinner('Đang khoanh vùng bởi Mask RCNN'):
                result = mrcnn.detect([mrcnn_input])[0]
            #st.success('Hoàn tất khoanh vùng')
            #merged = mrcnn_utils.process_to_visualize_output(mrcnn_input, result['masks'])
            #st.image(merged, caption='Output từ Mask RCNN')
            masks = unpad_image(result['masks'], (HEIGHT_TARGET, WIDTH_TARGET),
                    (HEIGHT, WIDTH))
            colors = random_colors(masks.shape[-1])
            num_ins_detect = str(result['masks'].shape[-1])
            st.write('Số lượng tế bào được tìm thấy bởi Mask RCNN: **{}**'.format(num_ins_detect))
            mrcnn_overlay = unpad_image(mrcnn_input, 
                    (HEIGHT_TARGET, WIDTH_TARGET), (HEIGHT, WIDTH)).copy() 
            for i in range(len(colors)):
                color = colors[i]
                mask = masks[..., i]
                mrcnn_overlay = apply_mask(mrcnn_overlay, mask, color, alpha=0.4)
            st.image(mrcnn_overlay)
        with col_cp:
            st.subheader('Output từ Cellpose')
            with st.spinner('Khởi tạo Cellpose và Size Model'):
                cp_model, sz_model = cellpose_utils.init_models()
            #st.success('Hoàn tất khởi tạo Cellpose và Size Model')
    
            with st.spinner('Đang khoanh vùng bởi Cellpose và Size Model'):
                pred_diam, _ = sz_model.eval(img, channels=[0, 0])
                mask, flow, _ = cp_model.eval(img, channels=[0, 0],
                diameter=pred_diam, augment=True)
            num_ins_detect = str(mask.max())
            st.write('Số lượng tế bào được tìm thấy bởi Cellpose: **{}**'.format(num_ins_detect))
            #st.success('Hoàn tất khoanh vùng')
            #merged = merge_image_and_mask(img, mask)
            #st.image(merged, caption='Output từ Cellpose'),
            overlay = mask_overlay(img, mask)
            st.image(overlay)

    
