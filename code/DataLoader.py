import os
import random
import numpy as np
import cv2
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from PIL import Image

class DataLoader(Dataset):
    def __init__(self,
                 mean,
                 std,
                 image_folder,
                 mask_folder,
                 transform=None):
        self.img_mean = mean
        self.img_std = std
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.imgs = os.listdir(self.image_folder)
        self.masks = os.listdir(self.mask_folder)


    def img2tensor(self, img, dtype:np.dtype=np.float32):
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        # HWC -> CHW
        img = np.transpose(img,(2,0,1))
        return torch.from_numpy(img.astype(dtype, copy=False))

    def preprocess(self, img, mask):
        h, w, c = img.shape
        h_gt, w_gt = mask.shape
        assert h==h_gt, "Error"
        assert w==w_gt, "Error"

        if self.transform:
            img, mask = self.transform(img, mask)
        img = np.array(img)
        mask = np.array(mask)
        mask = mask[:,:,np.newaxis]

        img = self.img2tensor((img/255.0 - self.img_mean)/self.img_std)
        mask = self.img2tensor(mask)
        return img, mask

    def __len__(self):
        return len(os.listdir(self.image_folder))

    def __getitem__(self, item):
        img, mask = self.imgs[item], self.masks[item]
        img = cv2.imread(os.path.join(self.image_folder, img), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_folder, mask), cv2.IMREAD_GRAYSCALE)
        img, mask = self.preprocess(img, mask)
        return img, mask

    # def __call__(self):
    #     img_names = os.listdir(self.image_folder)
    #     if self.shuffle:
    #         random.shuffle(img_names)
    #     for img_name in img_names:
    #         img = cv2.imread(os.path.join(self.image_folder, img_name), cv2.IMREAD_COLOR)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         mask = cv2.imread(os.path.join(self.mask_folder, img_name), cv2.IMREAD_GRAYSCALE)
    #         img, mask = self.preprocess(img, mask)
    #         yield img, mask

if __name__ == '__main__':
    print('test')