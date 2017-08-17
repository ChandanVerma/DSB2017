import math
import os
import numpy as np
import pandas as pd
import cv2
from glob import glob

label_df = pd.read_csv('D:/ImageAnalytics/stage1_labels.csv')
data_dir = 'F:/pred_stage1/'
rez_dir = 'D:/ImageAnalytics/resized_images/'

os.chdir(data_dir)
g = glob('*.npy')
patients = [patient.replace(".npy", "") for patient in g]

train_ids = [patient for patient in patients if patient in label_df["id"].values]
test_ids = [patient for patient in patients if patient not in label_df["id"].values]

IMG_SIZE_PX = 64
SLICE_COUNT = 20
hm_slices = 20

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def save_array(path, arr):
    np.save(path, arr)
    

def load_array(path):
    return np.load(path)


def mean(a):
    return sum(a) / len(a)

    
for patient in train_ids:
	slices = load_array(data_dir + "{}.npy".format(patient))
	new_slices = []
	slices = [cv2.resize(np.array(each_slice),(IMG_SIZE_PX,IMG_SIZE_PX)) for each_slice in slices]
	
	chunk_sizes = math.ceil(len(slices) / hm_slices)
	for slice_chunk in chunks(slices, chunk_sizes):
		slice_chunk = list(map(mean, zip(*slice_chunk)))
		new_slices.append(slice_chunk)
	
	if len(new_slices) == hm_slices-1:
		new_slices.append(new_slices[-1])
	
	if len(new_slices) == hm_slices-2:
		new_slices.append(new_slices[-1])
		new_slices.append(new_slices[-1])
	
	if len(new_slices) == hm_slices+2:
		new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
		del new_slices[hm_slices]
		new_slices[hm_slices-1] = new_val
		
	if len(new_slices) == hm_slices+1:
		new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
		del new_slices[hm_slices]
		new_slices[hm_slices-1] = new_val
	
	np.save(rez_dir + 'train/' + '{}.npy'.format(patient),np.array(new_slices))
	

   
for patient in test_ids:
	slices = load_array(data_dir + "{}.npy".format(patient))
	new_slices = []
	slices = [cv2.resize(np.array(each_slice),(IMG_SIZE_PX,IMG_SIZE_PX)) for each_slice in slices]
	
	chunk_sizes = math.ceil(len(slices) / hm_slices)
	for slice_chunk in chunks(slices, chunk_sizes):
		slice_chunk = list(map(mean, zip(*slice_chunk)))
		new_slices.append(slice_chunk)
	
	if len(new_slices) == hm_slices-1:
		new_slices.append(new_slices[-1])
	
	if len(new_slices) == hm_slices-2:
		new_slices.append(new_slices[-1])
		new_slices.append(new_slices[-1])
	
	if len(new_slices) == hm_slices+2:
		new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
		del new_slices[hm_slices]
		new_slices[hm_slices-1] = new_val
		
	if len(new_slices) == hm_slices+1:
		new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
		del new_slices[hm_slices]
		new_slices[hm_slices-1] = new_val
	
	np.save(rez_dir + 'test/' + '{}.npy'.format(patient),np.array(new_slices))
	

