# Sartorius-Cell-Instance-Segmentation

Use Deep Learning models to segment cells in microscopy image.

The dataset is downloaded from: https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data

**Sample image in this dataset**:

![Microscopy image sample in dataset](https://storage.googleapis.com/kagglesdsdata/competitions/30201/2750748/train/0030fd0e6378.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20220711%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220711T081747Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=6bed3ab9073f9a3c42c1749bc9f80aa10c8b431becf5a2e18a937508f27d0c3146973ba61f90e60be18ee8f658dc82fb8460833a5f12b81b5a9275a2428418e54d7df3bee51877365d59b76560d68c5feec359047c6a1a911e832c6f8c0021e3fc891c0602f613411f5fb623510f2c9bdfbe148c2d1b9273c2425e28b1f0f4265e4442a77b9ca88becb8be7b76d917ec624ac2d27d758c807da2cf5f8344b321bf914857db2c9d471d1ed9ef7fe9240480366226c4b2bad925885e1fbcf6f6194f9107f19123b9bb56064fb8f3d1f8d4a23e9049a1d1df318ae2f0ec8558a02581f16b358228eed40e2fe94f7f2e91ebbc1bbf849f62b6fb050cfa7a546d6486)

# Methods

**Models:** 
1. Mask RCNN
2. Cellpose

**Techniques:** 
1. Mosaic
2. Add extra data for the `SH-SY5Y` cell line from LIVECell dataset which is the predecessor of this dataset
3. Data Augmentation (Flip left/right, Flip up/down, Crop, Add noise, Rotation
4. L2 Regularization
5. Training Size Model (Cellpose's assistant model)

# Repository structure

## 📂 Browser folder
`./browser/` folder stores all demo files which use Mask RCNN or/and Cellpose to detect cells in microscopy image on web.

Using command `streamlit run main.py` to start server.
```
.
├── browser
│   ├── cellpose_utils.py
│   ├── main.py
│   ├── mrcnn_utils.py
│   └── utils.py
```

## 📂 Models folder
Mask RCNN and Cellpose packages are stored in `./models/` folder.
```
.
├── models
│   ├── Mask_RCNN
│   ├── cellpose
```

## 📂 Requirement folder
`./requirement/cellpose-requirement.txt` contains all packages used by Cellpose and 
`./requirement/mrcnn-requirement.txt` contains all packages used by Mask RCNN.
```
.
├── requirement
│   ├── cellpose-requirement.txt
│   └── mrcnn-requirement.txt
```

## 📂 Technique folder

`./technique/` folder stores all files demo of those techniques used in this project

```
.
├── technique
│   ├── augmentation
│   ├── helper_package
│   ├── livecell prepare.ipynb
│   ├── mini mask.ipynb
│   └── mosaic
```

## 📂 Train-infer-model folder

All `.ipynb` files which used to train and test model are stored in `./train-infer-model/` folder.

```
.
├── train-infer-model
│   ├── cellpose
│   ├── data
│   ├── mask_rcnn
│   └── performance
```

# Demo

Using `streamlit` framework to demo on website

...

# Reference

Mask RCNN: https://github.com/leekunhee/Mask_RCNN

Cellpose: https://github.com/MouseLand/cellpose

Streamlit: https://streamlit.io/


