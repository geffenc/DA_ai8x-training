import importlib

import os
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import distiller.apputils as apputils

import sys
sys.path.insert(0, 'models/')
sys.path.insert(1, 'distiller/')
sys.path.insert(2, 'datasets/')

from cats_and_dogs import *

mod = importlib.import_module("cat-dog_net")

import ai8x

class Args:
    def __init__(self, act_mode_8bit):
        self.act_mode_8bit = act_mode_8bit
        self.truncate_testset = False

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

args = Args(act_mode_8bit=False)
data_path = "../Datasets/cats_and_dogs/"

train_set, test_set = cats_and_dogs_get_datasets((data_path, args), load_train=True, load_test=True)

test_set.visualize_batch()