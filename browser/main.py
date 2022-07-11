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

from utils import unpad_image, merge_image_and_mask
import  mrcnn_utils 
import cellpose_utils

######################################
#       DEFINE CONSTANCE 
######################################

HEIGHT, WIDTH = 520, 704
HEIGHT_TARGET, WIDTH_TARGET = 576, 704
IMG_DIR = '../sartorius-cell-instance-segmentation/train'

######################################
#   BROWSER  
######################################

st.set_page_config(layout='wide')

st.title('Phát hiện và khoanh vùng tế bào')

# Selected box
option = st.selectbox('Chọn mô hình dự đoán', ('Mask RCNN', 'Cellpose', 'Cả hai mô hình'))
st.write('Bạn chọn: ', option)

# File uploader
uploaded_file = st.file_uploader("Chọn ảnh")
if uploaded_file is not None:
    # Load image input
    file_name = uploaded_file.name
    file_path = os.path.join(IMG_DIR, file_name)
    img = imageio.imread(file_path)

    # Show input
    st.image(img, caption='Input')

# Button
if st.button('Khoành vùng tế bào'):
    if option == 'Mask RCNN':
        with st.spinner('Khởi tạo mô hình Mask RCNN'):
            mrcnn = mrcnn_utils.init_and_load_weight()
            img = mrcnn_utils.prepare_input(img)
        st.success('Hoàn tất khởi tạo Mask RCNN')

        with st.spinner('Đang khoanh vùng bởi Mask RCNN'):
            result = mrcnn.detect([img])[0]
        st.success('Hoàn tất khoanh vùng')
        
        merged = mrcnn_utils.process_to_visualize_output(img, result['masks'])
        st.image(merged, caption='Output từ Mask RCNN')

    elif option == 'Cellpose':
        with st.spinner('Khởi tạo Cellpose và Size Model'):
            cp_model, sz_model = cellpose_utils.init_models()
        st.success('Hoàn tất khởi tạo Cellpose và Size Model')

        with st.spinner('Đang khoanh vùng bởi Cellpose và Size Model'):
            pred_diam, _ = sz_model.eval(img, channels=[0, 0])
            mask, flow, _ = cp_model.eval(img, channels=[0, 0],
                                            diameter=pred_diam, augment=True)
        st.success('Hoàn tất khoanh vùng')

        merged = merge_image_and_mask(img, mask)
        st.image(merged, caption='Output từ Cellpose')

    elif option == 'Cả hai mô hình':
        col_mrcnn, col_cp = st.columns(2)

        with col_mrcnn:
            with st.spinner('Khởi tạo mô hình Mask RCNN'):
                mrcnn = mrcnn_utils.init_and_load_weight()
                mrcnn_input = mrcnn_utils.prepare_input(img)
            st.success('Hoàn tất khởi tạo Mask RCNN')

            with st.spinner('Đang khoanh vùng bởi Mask RCNN'):
                result = mrcnn.detect([mrcnn_input])[0]
            st.success('Hoàn tất khoanh vùng')
        
            merged = mrcnn_utils.process_to_visualize_output(mrcnn_input, result['masks'])
            st.image(merged, caption='Output từ Mask RCNN')

        with col_cp:
            with st.spinner('Khởi tạo Cellpose và Size Model'):
                cp_model, sz_model = cellpose_utils.init_models()
            st.success('Hoàn tất khởi tạo Cellpose và Size Model')

            with st.spinner('Đang khoanh vùng bởi Cellpose và Size Model'):
                pred_diam, _ = sz_model.eval(img, channels=[0, 0])
                mask, flow, _ = cp_model.eval(img, channels=[0, 0],
                                            diameter=pred_diam, augment=True)
            st.success('Hoàn tất khoanh vùng')

            merged = merge_image_and_mask(img, mask)
            st.image(merged, caption='Output từ Cellpose')








    





