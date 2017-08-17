# Data Science Bowl 2017 Solution (No Transfer learning)
This repo explains us how to build a Convolutional neural network in Tensorflow for Medical Imaging.
The Solution is for Kaggle competition Data Science Bowl 2017.

# Problem statement: Build an Algorithm that is able to classify a given dicom image from CT modality into cancerous or non-cancerous.

The solution doen not use any external datasets like LIDC and purely based on the dataset provided by kaggle.
It also doesnt use any pretrained model like Unet, ResNet etc. 
The intention behind this repo is to create a framework for Deep learning in Medical Imaging.

The code files are provided in the above section:

Files must be executed in the following order:

1. Preprocessing.py
2. data_resize.py
3. data_split.py
4. data_stacking.py
5. label_stack.py
6. Tensorflow_CNN.py

It is expected that all the required libraries must be installed namely Tensorflow, skimage etc.
All  the codes are written in python.
Machine details:
1. Windows 10
2. GTX 1060 6GB GPU,
3. Intel core i7 Processor 
4. 64 GB RAM
5. 2 TB HDD
