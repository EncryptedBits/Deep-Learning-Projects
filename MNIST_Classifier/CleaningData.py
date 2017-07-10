from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
#import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
#from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

#root_folder = '.'
#test_folders = "/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNSIT"
#train_folders = os.path.join(root_folder,'notMNIST_small')
train_folders = os.path.join("/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNIST","notMNIST_large")
test_folders = os.path.join("/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNIST","notMNIST_small")

print(train_folders)

def create_dataset(folder,min_no_of_images):
    #print folder
    images_list = os.listdir(folder)
    data_set = np.ndarray(shape=(len(images_list),image_size,image_size),dtype=np.float32)
    no_of_images = 0
    for image in images_list:
        try:
            image_file = os.path.join(folder,image)
            image_data = (ndimage.imread(image_file).astype(float) - 128)/256
            if image_data.shape != (image_size,image_size):
                raise Exception('Unexpected image size : %s'%str(image_data.shape))
            
            data_set[no_of_images,:,:] = image_data
            no_of_images += 1
        except IOError as e:
            print('Unable to read image : ',e,"<skipping>")
    data_set = data_set[:no_of_images,:,:]
    
    if no_of_images < min_no_of_images:
        raise Exception('Less cleaned images than expected : %d < %d' %(no_of_images,min_no_of_images))
    else:
        print("######### %d images cleaned ##########",no_of_images)
        
    print('Mean:', np.mean(data_set))
    print('Standard deviation:', np.std(data_set))
    
    return data_set


#/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNSIT/notMNIST_large
#/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNSIT/notMNIST_large_pkl
sml_pkl = "/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNIST/notMNIST_small_pkl/"
big_pkl = "/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNIST/notMNIST_large_pkl/"


def maybe_pickle(folders, min_no_of_images,typee):
    dataset_names = []
    root_fldr = next(os.walk(folders))[0]
    for folder in next(os.walk(folders))[1]:
        set_classname = folder + ".pickle"
        print(set_classname)
        dataset_names.append(set_classname)
        class_dataset = create_dataset(os.path.join(root_fldr,folder),min_no_of_images)
        try:
            if typee == "train":
                with open(big_pkl+set_classname,"wb") as f:
                    pickle.dump(class_dataset,f,pickle.HIGHEST_PROTOCOL)
            else:
                with open(sml_pkl+set_classname,"wb") as f:
                    pickle.dump(class_dataset,f,pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            print ("Unable to save & pickle data to ",set_classname," : ",e)
    return dataset_names

    

print("Traindata-Pickling Started")
train_datasets = maybe_pickle(train_folders, 45000,"train")
print("Testdata-Pickling Started")
test_datasets = maybe_pickle(test_folders, 1800,"test")