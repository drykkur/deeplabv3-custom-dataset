import os
import sys
import glob
#import tarfile
#import collections
import torch.utils.data as data
import shutil
import numpy as np
from torchvision import transforms
from collections import namedtuple
import json

from PIL import Image
from PIL import ImageOps

class CustomSegmentation(data.Dataset):
    carClass = namedtuple('CarPartClass', ['name', 'id', 'train_id', 'color'])
    classes = [
        carClass('background',   0,255,(0,0,0)),
        carClass('front door',   1, 0, (250,149,10)),
        carClass('rear door',    2,1,(249,249,10)),
        carClass('frame',    3,2,(10,248,250)),
        carClass('rqp',  4,3,(149,7,149)),
        carClass('trunk lid',    5,4,(5,249,9)),
        carClass('fender',   6,5,(20,19,249)),
        carClass('bumper',  7,6,(249,9,250))
        ]
    
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self,
                 image_dir,  
                 mask_dir,
                 target_type='semantic',
                 mode='fine',
                 transform=None):
        self.transform = transform
        #self.transform = transforms.Compose([
        #    transforms.Resize((256, 256))
        #    ])
        # A: checking if the jpg directory exists
        #print(image_dir)
        if not os.path.isdir(image_dir):
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Check the source image folder.')
        # A: checking if the annotation mask directory exists
        if not os.path.exists(mask_dir):
            raise RuntimeError("Segmentation not found or corrupted.")

        # A: aggregating all files in the image_dir
        file_list = os.path.join(os.path.join(image_dir, "*.*"))
        
        # A: using glob to get the image names, since they can have more than one extension
        file_list = glob.glob(file_list)
        self.mode = 'gtFine'
        image_list = []
        tar_list=[]
        for file in file_list:
            # A: filtering since the directory also contains the .txt files used for YOLO annotation
            if ".txt" not in file:
                # A: keeping only the filename and its extension, without the path
                file = file.split("\\")[-1]
                #file = file
                #print(file)
                #file = os.path.splitext(file)[0]
                image_list.append(file)
                tar_list.append(file.format(mode))
        newimg = []
        #for i in image_list:
        #    newimg.append(image_dir + '/' + i)
        #print(image_dir)
        self.images = [os.path.join(image_dir, x) for x in image_list]
        # A: masks are always .png.
        self.targets = [os.path.join(mask_dir, os.path.splitext(x)[0] + ".png") for x in tar_list]
        #assert (len(self.images) == len(self.masks)) 
        #print(newimg)

        
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is the image segmentation.
#         """
#         try:
#             img = Image.open(self.images[index]).convert('RGB')
#             # A: portrait mode jpg images need to be transposed, because they are saved in landscape
#             # and rotated by an exif tag  
#             img = ImageOps.exif_transpose(img)
#             target = Image.open(self.masks[index])
#             if self.transform is not None:
#                 img, target = self.transform(img, target)
#             # A: the target has to be encoded after the transform is applied. Since the transform also
#             # changes the target into a tensor (on top of resizing the image and whatnot), np.array is
#             # needed to pass it into the encode function.
#             return img, self.encode_target(np.array(target))
#         except Exception as e:
#             print(e)
#             print(self.images[index])
#             print(self.masks[index])
#             img = Image.open("/home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/datasets/dummy.png").convert('RGB')
#             img = ImageOps.exif_transpose(img)
#             target = Image.open("/home/jovyan/work/deeplab/DeepLabV3Plus-Pytorch/datasets/dummy.png")
#             if self.transform is not None:
#                 img, target = self.transform(img, target)
#             # A: the target has to be encoded after the transform is applied. Since the transform also
#             # changes the target into a tensor (on top of resizing the image and whatnot), np.array is
#             # needed to pass it into the encode function.
#             return img, self.encode_target(np.array(target))

    
    @classmethod
    def encode_target(cls, target):
        #print(cls.id_to_train_id[np.array(target)])
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        print(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)