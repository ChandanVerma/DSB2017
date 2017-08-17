import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
import glob

def save_array(path, arr):
    np.save(path, arr)
    

def load_array(path):
    return np.load(path)

#%%

valid_ids = [id.replace(".npy", "") for id in os.listdir('C:/Unet/resized/valid/')]
data_dir = 'D:/Projects/Kaggle/DataScienceBowl/stage1/'
patients = os.listdir(data_dir)
labels = pd.read_csv('D:/Projects/Kaggle/DataScienceBowl/stage1_labels.csv', index_col=0)
rez_dir = 'C:/Unet/resized/'
valid_label = []
for num,patient in enumerate(valid_ids):
    if num % 100 == 0:
        print(num)
    try:
        label = labels.get_value(patient, 'cancer')
        #img_data = load_array('C:/Unet/resized/valid/{}.npy'.format(patient))
		 #img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        valid_label.append(np.float32(label))
    except KeyError as e:
        print('This is unlabeled data!')
        
np.save(rez_dir + 'valid_label.npy', valid_label)
	
#%%
train_ids = [id.replace(".npy", "") for id in os.listdir('C:/Unet/resized/train/')]
data_dir = 'D:/Projects/Kaggle/DataScienceBowl/stage1/'
patients = os.listdir(data_dir)
labels = pd.read_csv('D:/Projects/Kaggle/DataScienceBowl/stage1_labels.csv', index_col=0)
rez_dir = 'C:/Unet/resized/'
train_label = []
for num,patient in enumerate(train_ids):
    if num % 100 == 0:
        print(num)
    try:
        label = labels.get_value(patient, 'cancer')
        #img_data = load_array('C:/Unet/resized/train/{}.npy'.format(patient))
		 #img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        train_label.append(np.float32(label))
    except KeyError as e:
        print('This is unlabeled data!')
        
np.save(rez_dir + 'train_label.npy', train_label)
#%%


test_ids = [id.replace(".npy", "") for id in os.listdir('C:/Unet/resized/test/')]
data_dir = 'D:/Projects/Kaggle/DataScienceBowl/stage1/'
patients = os.listdir(data_dir)
labels = pd.read_csv('D:/Projects/Kaggle/DataScienceBowl/stage1_labels.csv', index_col=0)
rez_dir = 'C:/Unet/resized/'
test_label = []
for num,patient in enumerate(test_ids):
    if num % 100 == 0:
        print(num)
    try:
        label = 0
        #img_data = load_array('C:/Unet/resized/test/{}.npy'.format(patient))
		 #img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        test_label.append(np.float32(label))
    except KeyError as e:
        print('This is unlabeled data!')
        
np.save(rez_dir + 'test_label.npy', test_label)
	
