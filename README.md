# Project Title: Cancer Detection using Machine Learning

## Description:
This project addresses the binary image classification problem of identifying metastases in 96 x 96px digital histopathology images. Our aim is to achieve high accuracy using advanced machine learning techniques.

## Data Overview:
The dataset consists of 220k training images and 57k evaluation images, focusing on lymph nodes stained with hematoxylin and eosin. It's a subset of the PCam dataset, with duplicates removed. A positive label indicates the presence of tumor tissue in the center region (32 x 32px) of the image.

## Getting Started

1. Clone the repository.
2. Install necessary dependencies.
3. Follow instructions in the documentation for running the project.

## Python script 
* python main.py --dataset pathology --bias 0 --num_workers 20 --cmax 0 --num_rounds 150 --num_iters 10 --gpu 0 --aggregation fedsgd --exp train_patho  