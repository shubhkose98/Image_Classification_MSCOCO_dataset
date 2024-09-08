# Image Classification on MS-COCO dataset
## Overview
In this project, we explore the implementation of Convolutional Neural Networks (CNNs) in PyTorch for image classification tasks. We utilize the MS-COCO (Microsoft Common Objects in COntext) a widely-used dataset for image classification and object detection tasks dataset, built a training and validation wrapper, and evaluated the results. The project demonstrates the process of building and training three different CNN architectures with varying layers such as convolutional, max pooling, and fully connected layers. The networks are trained and validated using a subset of the 2017 version of the COCO dataset. Finally, the performance of the models is compared by computing confusion matrices, which help in identifying how well each network classified the images.

## Dataset
The project uses the MS-COCO 2014 dataset for training and validation. The dataset consists of images from a variety of object categories commonly encountered in daily life.

How to Download the Dataset:
Install the COCO API:
```pip install pycocotools```

Use the COCO API to download the subset of images for selected categories.
The downloaded images are split into train/ and test/ folders, organized by category (e.g., motorcycle, dog, cake).
You can read more about MS-COCO and download the dataset from the official website: [MS-COCO Dataset](https://cocodataset.org/#download).

## Implementation

- CNN Architectures: 
Three different CNN architectures were designed, each with a unique combination of convolutional, max pooling, and linear layers.
Each network is implemented using PyTorch and trained on the COCO dataset.
We employed SkipBlock residual connections in some architectures to enhance learning by bypassing certain layers.
- Training & Validation: 
The models were trained and validated using the downloaded subset of the COCO dataset.
Performance was evaluated using confusion matrices to compare classification accuracy across different categories.

## Requirements
The PDF file attached contains all of the implementation documentation and libraries and datasets required. Also, please check the imports in the .ipynb or .py file.


