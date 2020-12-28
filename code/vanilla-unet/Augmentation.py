from torchvision import transforms
from PIL import Image

class Augmentation(object):
    def __init__(self, image_size):
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.RandomRotation(90),
                transforms.RandomResizedCrop(image_size, scale=(0.4, 1), ratio=(0.8, 1.2)),
            ]),
            transforms.ColorJitter(brightness=0.8,contrast=0.8)
        ])

    def __call__(self, img, mask):
        return self.augment(Image.fromarray(img)), self.augment(Image.fromarray(mask))