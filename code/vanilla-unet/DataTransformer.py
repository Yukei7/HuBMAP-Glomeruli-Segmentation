import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
from tqdm import tqdm

class DataTransformer(object):
    def __init__(self, mask_path='../train.csv', data_path='../train', size=512, reduce=2):
        # size of tiles
        self.size = size
        # scale
        self.reduce = reduce
        self.mask_path = mask_path
        self.data_path = data_path

    def __call__(self, img_output_path='../train_img', mask_output_path='../mask_img', *args, **kwargs):
        if not os.path.exists(img_output_path):
            os.mkdir(img_output_path)
        if not os.path.exists(mask_output_path):
            os.mkdir(mask_output_path)

        # For statistics
        x_tot, x_tot2 = [], []

        df_masks = pd.read_csv(self.mask_path).set_index('id')
        for index, encs in tqdm(df_masks.iterrows(), total=len(df_masks)):
            img = tiff.imread(os.path.join(self.data_path, index+'.tiff'))
            if len(img.shape) == 5:
                # Normal case: HWC
                # Weird case: 1,1,C,H,W
                img = np.transpose(img.squeeze(), (1,2,0))
            mask = self.enc2mask(encs, [img.shape[1], img.shape[0]])

            # add padding to make image dividable into tiles
            shape = img.shape
            nums = self.reduce*self.size
            # split into small block, then reduce the scale
            pad0 = (nums - shape[0]%nums) % nums
            pad1 = (nums - shape[1]%nums) % nums
            img = np.pad(img, [[pad0//2,pad0-pad0//2], [pad1//2,pad1-pad1//2], [0,0]],
                         constant_values=0)
            mask = np.pad(mask, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2]],
                          constant_values=0)

            # split the image and the mask into tiles
            img = cv2.resize(img, (img.shape[1]//self.reduce, img.shape[0]//self.reduce),
                             interpolation=cv2.INTER_AREA)
            # Transform into small block
            # W,W_s,H,H_s,C
            img = img.reshape(img.shape[0]//self.size, self.size, img.shape[1]//self.size, self.size, 3)
            img = img.transpose(0,2,1,3,4).reshape(-1, self.size, self.size, 3)

            mask = cv2.resize(mask, (mask.shape[1] // self.reduce, mask.shape[0]//self.reduce),
                              interpolation=cv2.INTER_NEAREST)
            mask = mask.reshape(mask.shape[0]//self.size, self.size, mask.shape[1]//self.size, self.size)
            mask = mask.transpose(0,2,1,3).reshape(-1, self.size, self.size)

            for i,(im,m) in enumerate(zip(img,mask)):
                # Saturation threshold
                s_thresh = 40
                # Minimum number threshold
                p_thresh = 200*self.size//256
                hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                if (s>s_thresh).sum() <= p_thresh or im.sum() < p_thresh:
                    continue

                x_tot.append((im/255.0).reshape(-1,3).mean(0))
                x_tot2.append(((im/255.0)**2).reshape(-1,3).mean(0))

                im_path = os.path.join(img_output_path, str(index)+'_'+str(i)+'.png')
                m_path = os.path.join(mask_output_path, str(index)+'_'+str(i)+'.png')
                im = cv2.imencode('.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1].tofile(im_path)
                m = cv2.imencode('.png', m)[1].tofile(m_path)

        im_avg = np.array(x_tot).mean(0)
        im_std = np.sqrt(np.array(x_tot2).mean(0) - im_avg**2)
        print('avg: {}, std: {}'.format(im_avg, im_std))


    def enc2mask(self, encs, shape):
        # Converting encoding to mask
        # See: RLE Encoding
        # Encoding is formatted as [RLEEnc, length, RLEEnc, length ...]
        # Since it's Column-major here, length is add to the column.
        # Split with space
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for i,enc in enumerate(encs):
            if isinstance(enc, np.float) and np.isnan(enc):
                continue
            enc_split = enc.split()
            for idx in range(len(enc_split)//2):
                start = int(enc_split[2*idx]) - 1
                length = int(enc_split[2*idx+1])
                img[start:start+length] = 1 + i
        # Note that encoding is Column-major and in Python it's Row-major
        return img.reshape(shape).T


    def mask2enc(self, mask):
        pixels = mask.T.flatten()

        pixels[0] = 0
        pixels[-1] = 0
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

if __name__ == '__main__':
    DT = DataTransformer(size=256, reduce=4)
    DT(img_output_path='../train_img_256_r4', mask_output_path='../mask_img_256_r4')
