###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Datasets for classifying images
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image
import torch
import pandas as pd
import os
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import ai8x


'''
Dataset Class
Parameters:
  img_dir_path - Full path to directory with the images for this dataset.
                 This assumes that the subdirectories contain each class, 
                 only images are in these subdirectories, and that the
                 subdirectory basenames are the desired name of the object class.
                 i.e. dog/dog1.png, cat/cat1.png, etc.
  transform -    Specifies the image format (size, RGB, etc.) and augmentations to use
  normalize -    Specifies whether to make the image zero mean, unit variance
'''
class ClassificationDataset(Dataset):
    def __init__(self,img_dir_path,transform,normalize):
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.normalize = normalize

        print(self.img_dir_path)
        
        # collect img classes from dir names
        img_classes = next(os.walk(img_dir_path))[1]
        
        # generate a dictionary to map class names to integers idxs
        self.classes = {img_classes[i] : i for i in range(0, len(img_classes))}
        self.label_dict = {v: k for k, v in self.classes.items()}
        print(self.classes)
        
        # get all training samples/labels by getting absolute paths of the images in each subfolder
        self.imgs = [] # absolute img paths (all images)
        self.labels = [] # integer labels (all labels in corresponding order)

        i = 0 # index into dataset lists

        # iterate through the dataset directory tree
        for idx, path_obj in enumerate(os.walk(img_dir_path)):
            # each execution of this inner loop is for each subdirectory
            if idx > 0: # don't include files in the top folder (subfolders are in the next itertion, idx > 0)
                for file in path_obj[2]: # path_obj[2] is list of files in the object class subdirectories
                    self.imgs.append(os.path.abspath(os.path.join(path_obj[0],file))) # want absolute path
                    self.labels.append(self.classes[os.path.basename(os.path.dirname(self.imgs[i]))]) # get label from directory name
                    i+=1

    # dataset size is number of images
    def __len__(self):
        return len(self.imgs)
    
    # how to get one sample from the dataset
    def __getitem__(self, idx):
        # attempt to load the image at the specified index
        try:
            img = Image.open(self.imgs[idx])
            tt = torchvision.transforms.ToTensor()
            tp = torchvision.transforms.ToPILImage()
            tt_img = tt(img)
            if(tt_img.size()[0] != 3):
                img = tp(tt_img.repeat(3, 1, 1))
            
            # apply any transformation
            if self.transform:
                img = self.transform(img)
                if self.normalize:
                    norm = torchvision.transforms.Normalize((torch.mean(img)),(torch.std(img)))
                    img = norm(img)
            
            # get the label
            label = self.labels[idx]
            
            # return the sample (img (tensor)), object class (int), and the path optionally
            return img, label, os.path.basename(self.imgs[idx])

        # if the image is invalid, show the exception
        except (ValueError, RuntimeWarning,UserWarning) as e:
            print("Exception: ", e)
            print("Bad Image: ", self.imgs[idx])
            exit()
    
    # Diaply the results of a forward pass for a random batch of 64 samples
    # if no model passed in, just display a batch with no predictions
    def visualize_batch(self,model=None,device=None):
        #import matplotlib
        #matplotlib.use('TkAgg')

        # create the dataloader
        batch_size = 64
        data_loader = DataLoader(self,batch_size,shuffle=True)

        # get the first batch
        (imgs, labels, paths) = next(iter(data_loader))
        #(imgs, labels) = next(iter(data_loader))
        imgs,labels = imgs.to(device), labels.to(device)
        preds = None

        # check if want to do a forward pass
        if model != None:
            preds = model(imgs)
        
        imgs,labels = imgs.to("cpu"), labels.to("cpu")
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        obj_classes = list(self.classes)
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        #fig.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j

                # create text labels
                text = str(labels[idx].item())
                if model != None:
                    text = "GT :" + obj_classes[labels[idx]]  + " P: ",obj_classes[preds[idx].argmax()]#", i=" +str(idxs[idx].item())
                
                # for normal forward pass use this line
                #ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))

                # for quantized forward pass use this line
                #print(imgs[idx].size(),torch.min(imgs[idx]))
                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)+1)/2)

                ax_array[i,j].set_title(text,color="white")
                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
        plt.savefig('plot.png')
        #print(paths)
        #plt.show()

    def viz_mispredict(self,wrong_samples,wrong_preds,actual_preds,img_names):
        wrong_samples,wrong_preds,actual_preds = wrong_samples.to("cpu"), wrong_preds.to("cpu"),actual_preds.to("cpu")
        
        # import matplotlib
        # matplotlib.use('TkAgg')
        obj_classes = list(self.classes)
        num_samples = len(wrong_samples)
        num_rows = int(np.floor(np.sqrt(num_samples)))

        if num_rows > 0:
            num_cols = num_samples // num_rows
        else:
            return
        print("num wrong:",num_samples, " num rows:",num_rows, " num cols:",num_cols)

        fig,ax_array = plt.subplots(num_rows,num_cols,figsize=(30,30))
        fig.subplots_adjust(hspace=1.5)
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i*num_rows+j
                sample = wrong_samples[idx]
                wrong_pred = wrong_preds[idx]
                actual_pred = actual_preds[idx]
                # Undo normalization
                sample = (sample.permute(1, 2, 0)+1)/2
                #text = "L: " + obj_classes[actual_pred.item()]  + " P:",obj_classes[wrong_pred.item()]#", i=" +str(idxs[idx].item())
                text = img_names[idx]
                
                # for normal forward pass use this line
                #ax_array[i,j].imshow(imgs[idx].permute(1, 2, 0))

                # for quantized forward pass use this line
                #print(imgs[idx].size(),torch.min(imgs[idx]))
                try:
                    if(ax_array.ndim > 1):
                        ax_array[i,j].imshow(sample)
                        ax_array[i,j].set_title(text,color="white")
                        ax_array[i,j].set_xticks([])
                        ax_array[i,j].set_yticks([])
                except:
                    print("exception")
                    print(ax_array.ndim)
                    print(sample)
                    return
        plt.savefig("incorrect.png")


# ================================= Helper functions for domain adaptation ===========================
# code derived from https://github.com/HX-idiot/FADA-Pytorch/blob/master/dataloader.py
# ====================================================================================================

# gets source samples/labels from source dataset as separate lists
def create_source_samples(source_data_path,args):
    train_set, test_set = cats_and_dogs_get_datasets((source_data_path, args), load_train=True, load_test=False,apply_transforms=False)
    n = len(train_set)
    X=torch.Tensor(n,3,128,128)
    Y=torch.LongTensor(n)

    inds=torch.randperm(len(train_set))
    for i,index in enumerate(inds):
        x,y,name=train_set[index]
        X[i]=x
        Y[i]=y
    return X,Y


# gets target samples/labels from target dataset where
# shot is the number of samples to get for each class. 
def create_target_samples(shot,target_data_path,args):
    # get the target dataset
    train_set, test_set = cats_and_dogs_get_datasets((target_data_path, args), load_train=True, load_test=False,apply_transforms=False)
    num_classes = 2

    X,Y=[],[]

    # list of counts to get equal amount per class, e.g. [2,2] --> 2 dog samples and 2 cat samples
    class_counts=num_classes*[shot]

    # keep getting items from the dataset until we have equal amount per class
    i=0
    while True:
        if len(X)==shot*num_classes:
            break
        x,y,name=train_set[i]
        if class_counts[y]>0:
            X.append(x)
            Y.append(y)
            class_counts[y]-=1
        i+=1

    assert (len(X)==shot*num_classes)
    return torch.stack(X,dim=0),torch.from_numpy(np.array(Y))


"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes
G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""
def create_groups(X_s,Y_s,X_t,Y_t,shot,seed=1):
    C = 2
    # shuffle order
    classes = torch.unique(Y_t)
    classes=classes[torch.randperm(len(classes))]

    # change seed so every epoch to randomize the source data while keeping target data the same
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    # mapping function: given a class this function will return a list
    # of indices for the corresponding class e.g. 0 --> [1,3]
    def s_idxs(c):
        # get a list of indices for the elements of class c
        idx=torch.nonzero(Y_s.eq(int(c)))
        # shuffle the list and flatten it, twice as many source samples
        return idx[torch.randperm(len(idx))][:shot*2].squeeze()
    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    # the result is a lists of lists: [0,1,2] --> [[1,3],[0,2],[4,5]]
    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    # stack the sublists into a matrix, i.e. each row is a class
    # [
    # C0  [1,3],
    # C1  [0,2],
    # C2  [4,5]
    # ]
    source_matrix=torch.stack(source_idxs)
    target_matrix=torch.stack(target_idxs)

    # now sample from matrices to get the four groupings
    G1, G2, G3, G4 = [], [] , [] , []
    Y1, Y2 , Y3 , Y4 = [], [] ,[] ,[]


    for i in range(C):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]],X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]],Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i%C][j]],X_s[source_matrix[(i+1)%C][j]]))
            Y3.append((Y_s[source_matrix[i % C][j]], Y_s[source_matrix[(i + 1) % C][j]]))
            G4.append((X_s[source_matrix[i%C][j]],X_t[target_matrix[(i+1)%C][j]]))
            Y4.append((Y_s[source_matrix[i % C][j]], Y_t[target_matrix[(i + 1) % C][j]]))

    groups=[G1,G2,G3,G4]
    groups_y=[Y1,Y2,Y3,Y4]

    # make sure we sampled enough samples
    for g in groups:
        assert(len(g)==shot*C)
    return groups,groups_y


def sample_groups(X_s,Y_s,X_t,Y_t,shot,seed=1):
    print("Sampling groups")
    return create_groups(X_s,Y_s,X_t,Y_t,shot,seed=seed)

'''Function to get the datasets'''
def cats_and_dogs_get_datasets(data, load_train=True, load_test=True,apply_transforms=True):
    (data_dir, args) = data

    train_dataset = None
    test_dataset = None

    # transforms for training
    if load_train and apply_transforms:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            #transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.65,1.35),saturation=(0.65,1.35),contrast=(0.65,1.35)),#,hue=(-0.1,0.1)),
            #transforms.RandomGrayscale(0.15),
            transforms.RandomAffine(degrees=20,translate=(0.25,0.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = ClassificationDataset(os.path.join(data_dir,"train"),train_transform,normalize=False)

    elif load_train and not apply_transforms:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = ClassificationDataset(os.path.join(data_dir,"train"),train_transform,normalize=False)

    else:
        train_dataset = None

    # transforms for test, validatio --> convert to a valid tensor
    if load_test:
        test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        test_dataset = ClassificationDataset(os.path.join(data_dir,"test"),test_transform,normalize=False)

    else:
        test_dataset = None
    
    return train_dataset, test_dataset


def imagenet10_get_datasets(data, load_train=True, load_test=True,apply_transforms=True):
    (data_dir, args) = data

    train_dataset = None
    test_dataset = None

    # transforms for training
    if load_train and apply_transforms:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            #transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.65,1.35),saturation=(0.65,1.35),contrast=(0.65,1.35)),#,hue=(-0.1,0.1)),
            #transforms.RandomGrayscale(0.15),
            transforms.RandomAffine(degrees=10,translate=(0.2,0.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = ClassificationDataset(os.path.join(data_dir,"train"),train_transform,normalize=False)

    elif load_train and not apply_transforms:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = ClassificationDataset(os.path.join(data_dir,"train"),train_transform,normalize=False)

    else:
        train_dataset = None

    # transforms for test, validatio --> convert to a valid tensor
    if load_test:
        test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        test_dataset = ClassificationDataset(os.path.join(data_dir,"test"),test_transform,normalize=False)

    else:
        test_dataset = None
    
    return train_dataset, test_dataset


def office5_get_datasets(data, load_train=True, load_test=True,apply_transforms=True):
    (data_dir, args) = data

    train_dataset = None
    test_dataset = None

    # transforms for training
    if load_train and apply_transforms:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            #transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.65,1.35),saturation=(0.65,1.35),contrast=(0.65,1.35)),#,hue=(-0.1,0.1)),
            #transforms.RandomGrayscale(0.15),
            transforms.RandomAffine(degrees=10,translate=(0.2,0.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = ClassificationDataset(os.path.join(data_dir,"train"),train_transform,normalize=False)

    elif load_train and not apply_transforms:
        train_transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        train_dataset = ClassificationDataset(os.path.join(data_dir,"train"),train_transform,normalize=False)

    else:
        train_dataset = None

    # transforms for test, validatio --> convert to a valid tensor
    if load_test:
        test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])
        test_dataset = ClassificationDataset(os.path.join(data_dir,"test"),test_transform,normalize=False)

    else:
        test_dataset = None
    
    return train_dataset, test_dataset

datasets = [
    {
        'name': 'cats_and_dogs',
        'input': (3, 128, 128),
        'output': ('dog', 'cat'),
        'loader': cats_and_dogs_get_datasets,
    },
]