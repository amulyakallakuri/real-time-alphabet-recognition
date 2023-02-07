import os
from tkinter.tix import DirTree
import h5py
import numpy as np
from PIL import Image
import glob
import re 
import collections
from sklearn.model_selection import train_test_split
split_ratio = 0.3#train/val = 10

data_dir =  r'./dataset/archive/asl_alphabet_train'
val_dir = r'./dataset/archive/asl_alphabet_val'
if os.path.exists(val_dir) ==False:
    os.mkdir(val_dir)

image_dirs = glob.glob(data_dir+'/**/*.jpg',recursive=True)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
dir_dict = {c:[] for  c in classes}
for image in image_dirs:
    label = re.search('[a-zA-Z]*',image.split('/')[-1]).group()
    dir_dict[label].append(image)
# val_dict = {}
# train_dict = {}
for key in dir_dict:
    key_img_paths = dir_dict[key]
    val_key_dir = os.path.join(val_dir,key)
    if os.path.exists(val_key_dir) ==False:
        os.mkdir(val_key_dir)
    train, val = train_test_split(key_img_paths,test_size = split_ratio)
    print(len(val))
    for img in val:
        os.rename(img, os.path.join(val_key_dir,img.split('/')[-1]))

    # val[key]=val
    # train[key]=train


    