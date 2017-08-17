import os
import numpy as np
import pandas as pd

from glob import glob

label_df = pd.read_csv('D:/ImageAnalytics/stage1_labels.csv')
data_dir = 'D:/ImageAnalytics/resized_images/train/'
print(label_df.head)

#%cd data_dir

os.chdir(data_dir)
g = glob('*.npy')
patients = [patient.replace(".npy", "") for patient in g]

#patients[:10]
#test_ids = [patient for patient in patients if patient not in label_df["id"].values]
#print(len(test_ids))
#test_ids[:5]
#for id in test_ids:
#    fn = "{}.npy".format(id)
#    os.rename(fn,data_dir + 'test/' + fn)

g = glob('*.npy')
shuf = np.random.permutation(g)

for i in range(200):
    os.rename(shuf[i],'D:/ImageAnalytics/resized_images/valid/' + shuf[i])
