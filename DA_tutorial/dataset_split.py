'''
dataset_split.py
This file splits a dataset into training
and test folders. The training data gets
split into validation partitions during
training.
'''

import os
from sklearn.model_selection import train_test_split
import shutil

'''
This function will split a folder which contains
subfolders of classes into train and test folders
with corresponding class subfolders.
Parameters:
  folder_path - the path to the folder with the data
  
  test_percent - the desired percent of the data used for testing.
                  Train data will be (1 - test_percent)
'''
def split_dataset(folder_path,test_percent):
    # collect img classes from dir names
    img_classes = next(os.walk(folder_path))[1]
    
    # create the train and test folders
    os.mkdir(os.path.join(folder_path,"train"))
    os.mkdir(os.path.join(folder_path,"test"))
    
    # get partitions of all the classes for train and test
    for img_class in img_classes:
        train_files, test_files = split_folder(os.path.join(folder_path,img_class), test_percent)
        
        # copy training files to training directory
        os.mkdir(os.path.join(folder_path,"train",img_class))
        for img_file in train_files:
            shutil.copy(os.path.join(folder_path,img_class,img_file),os.path.join(folder_path,"train",img_class))
        
        # copy test files to test directory
        os.mkdir(os.path.join(folder_path,"test",img_class))
        for img_file in test_files:
            shutil.copy(os.path.join(folder_path,img_class,img_file),os.path.join(folder_path,"test",img_class))
            
        # delete the old folder (commented out because we want to keep original file structure)
        #os.rmdir(os.path.join(folder_path,img_class))
        

'''
This function will split a folder into training
and test data given a testing percent. This assumes
the folder contains a single class of data.
Parameters:
  folder_path - the path to the folder with the images
  
  test_percent - the desired percent of the data used for testing.
                  Train data will be (1 - test_percent)
'''
def split_folder(folder_path,test_percent):
    # get the file names
    imgs = os.listdir(folder_path)
    train,test = train_test_split(imgs,test_size=test_percent)
    return train,test
    
    
    
if __name__ == "__main__":
    folder_path = "asl/source/"
    test_split = 0.10 # 10% test, 90% train 
    split_dataset(folder_path,test_split) 