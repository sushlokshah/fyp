import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

def read_dataset(data_root, mode):
    """
    Read dataset from data_root.
    """
    if mode == 'train':
        # list all sequence folders in train dir
        path = os.path.join(data_root, 'train')
        seq_list = os.listdir(path)
        
    elif mode == 'test':
        path = os.path.join(data_root, 'test')
        seq_list = os.listdir(path)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
    
    dataset = {}
    seq_lens = {}
    for seq in seq_list:
        seq_root = os.path.join(path, seq)
        img_list = os.listdir(seq_root)
        img_list = [os.path.join(seq_root, img) for img in img_list]
        img_list.sort()
        
        dataset[seq] = img_list
        seq_lens[seq] = len(img_list)
    
    return dataset, seq_lens
    


class Gopro(Dataset):
    """_summary_

    Args:
        data_root, transform, seq_len = [5,20], rate=[1, 10],train=True, mode='train'
    """
    def __init__(self,args,transform,mode):
        # Set arguments
        if mode =="train":
            self.data_root = args.data_root_train   
            self.max_seq_len = args.seq_len_train[-1]
            self.train = args.train
            self.transform = transform
            self.mode = 'train'
        
        elif mode =="test":
            self.data_root = args.data_root_test
            self.max_seq_len = args.seq_len_test[-1]
            self.train = args.test
            self.transform = transform
            self.mode = 'test'

        # Read dataset
        self.raw_data, self.seq_lens = read_dataset(self.data_root, self.mode)
        self.seq = list(self.raw_data.keys())
        self.total_imgs = 0
        for i in range(len(self.seq)):
            self.total_imgs += self.seq_lens[self.seq[i]] 
        # print(self.total_imgs)
            
    def __len__(self):
        return self.total_imgs
        
    def __getitem__(self, idx):
        seq_num = 0
        while(idx - self.seq_lens[self.seq[seq_num]] >= 0):
            idx = idx - self.seq_lens[self.seq[seq_num]]
            seq_num += 1
        seq = self.seq[seq_num]
        # print(seq)
        
        self.image_list = []
        
        if(idx + self.max_seq_len >= self.seq_lens[seq]):
            idx = self.seq_lens[seq] - self.max_seq_len - 1
        
        blurry_image = 0    
        for i in range(self.max_seq_len):
            # read image from path
            img_path = self.raw_data[seq][idx + i]
            img = Image.open(img_path)
            blurry_image = blurry_image + np.asarray(img)/(self.max_seq_len)
            self.image_list.append(self.transform(img).unsqueeze(0))
    
        self.blurry_image = Image.fromarray(np.uint8(blurry_image))
        
        # Pack data
        data = {
            'length': torch.tensor(self.max_seq_len),
            'blur': self.transform(self.blurry_image),
            'gen_seq': torch.cat(self.image_list, dim=0)
        }
        # print("inter_frame_distance: ", torch.tensor(self.inter_frame_distance))
        # print("length: ", torch.tensor(self.seq_gen))
        # print("blur: ", self.transform(self.blurry_image).shape)
        # print("gen_seq: ", len(self.image_list))
        
        return data


def get_transform(args,mode):
    if mode == "train":
        augmentation = []
        # if resize is present in the dictionary, resize the image
        if 'resize' in args.training_augmentations:
            augmentation.append(transforms.Resize(args.training_augmentations['resize']))
        
        if 'color_jitter' in args.training_augmentations:
            brightness = args.training_augmentations['color_jitter'][0]
            contrast = args.training_augmentations['color_jitter'][1]
            saturation = args.training_augmentations['color_jitter'][2]
            hue = args.training_augmentations['color_jitter'][3]
            augmentation.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
            
        if 'random_flip' in args.training_augmentations:
            augmentation.append(transforms.RandomHorizontalFlip())
        
        if 'random_rotation' in args.training_augmentations:
            augmentation.append(transforms.RandomRotation(args.training_augmentations['random_rotation']))
        
        augmentation.append(transforms.ToTensor())
        augmentation.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # print(augmentation)
        
        # return transform
        transform = transforms.Compose(augmentation)
        return transform
    
    elif mode == "test":
        augmentation = []
        # if resize is present in the dictionary, resize the image
        if 'resize' in args.test_augmentations:
            augmentation.append(transforms.Resize(args.test_augmentations['resize']))
        
        augmentation.append(transforms.ToTensor())
        augmentation.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # print(augmentation)
        
        # return transform
        transform = transforms.Compose(augmentation)
        return transform
        