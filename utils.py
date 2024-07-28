import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, albumentations=False):
        self.transform = transform
        self.albumentations = albumentations

    def __call__(self, x):
        if self.albumentations:
            return [self.transform(image=np.array(x)), self.transform(image=np.array(x))]
        else:
            return [self.transform(x), self.transform(x)]

def save_vit_model(model, save_path):
    state = {
        'model': model.state_dict(),
    }
    torch.save(state, save_path)
    print(f'=> save the model..')
    print('='*100)


def save_cat_model(cat, tr_acc, save_path):
    state = {
        'model': cat.state_dict(),
        'train_acc': tr_acc,
    }
    torch.save(state, save_path)
    print(f'=> save the model..')
    print('=' * 100)