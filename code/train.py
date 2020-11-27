import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam

import argparse
import random
import os
import numpy as np

from DataLoader import DataLoader
from Augmentation import Augmentation
from DiceLoss import DiceLoss

from unetpp import Unetpp
from utils import AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='unetpp')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--image_folder', type=str, default='../train_img_256_r4')
parser.add_argument('--mask_folder', type=str, default='../mask_img_256_r4')
parser.add_argument('--checkpoint_folder', type=str, default='../checkpoint_256_r4')
parser.add_argument('--save_freq', type=int, default=10)

args = parser.parse_args()

def train(dataloader, model, criterion, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, (im,mask) in enumerate(dataloader):
        im = im.cuda()
        mask = mask.cuda()
        pred = model(im)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = im.shape[0]
        train_loss_meter.update(loss.cpu().detach().numpy(), n)
        print(f"Epoch [{epoch:03d}/{args.num_epochs:03d}], " +
              f"Step[{batch_id:04d}/{total_batch:04d}], " +
              f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg

def main():
    SIZE = 256
    # set env
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # define dataloader
    train_aug = Augmentation(image_size=SIZE)
    # 256x256,r4
    # avg: [0.65459856 0.48386562 0.69428385], std: [0.15167958 0.23584107 0.13146145]
    img_avg = np.array([0.65459856, 0.48386562, 0.69428385])
    img_std = np.array([0.15167958, 0.23584107, 0.13146145])
    train_data = DataLoader(img_avg, img_std,
                            image_folder=args.image_folder,
                            mask_folder=args.mask_folder,
                            transform = train_aug)
    total_batch = int(len(train_data) / args.batch_size)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

    # create model
    if args.net == 'unetpp':
        model = Unetpp(in_channels=3, n_classes=1)
    else:
        raise ValueError("No such model!")
    model.cuda()

    # define criterion and optimizer
    criterion = DiceLoss
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(1, args.num_epochs+1):
        train_loss = train(train_dataloader,
                           model,
                           criterion,
                           optimizer,
                           epoch,
                           total_batch)
        print(f'---- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f}')

        if epoch % args.save_freq == 0:
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}--Epoch-{epoch}-Loss-{train_loss}")
            torch.save(model.state_dict(), model_path+'.pkl')

        # validation()
    model_path = os.path.join(args.checkpoint_folder, f"{args.net}")
    torch.save(model.state_dict(), model_path+'.pkl')
    print("Model saved.")


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_seed(42)
    if not os.path.exists(args.checkpoint_folder):
        os.mkdir(args.checkpoint_folder)
    main()
