import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms

class DogCat(data.Dataset):

    def __init__(self, root, transform=None, mode=None):
        """
        target: obtain all the address of imgs, and split them
        mode ∈ ['train', 'test', 'val']
        """
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # print(imgs)

        if self.mode == 'test':
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # split train, val, the propotion is 7:3
        if self.mode == 'test': self.imgs = imgs
        elif self.mode == 'train': self.imgs = imgs[:int(0.7*imgs_num)]
        elif self.mode == 'val': self.imgs = imgs[int(0.7*imgs_num):]

        if transform is None:
            # data transformarion will be different for train mode and val mode
            normalization = transforms.Normalize(mean=[0.485, 0.456,0.406],
                                                 std = [0.299, 0.224, 0.225])
            
            # no need to do data augmentation for val and test
            if self.mode == 'test' or self.mode=='val':
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalization
                ])
            # train data needs augmentation
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalization
                ])
    
    def __getitem__(self, index):
        """
        return data of img
        for val data, return id. 1000.jpg->1000
        """
        img_path = self.imgs[index]
        if self.mode == 'test':
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])  # actually the path of img
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label
    
    def __len__(self):
        """
        return the num of imgs
        """
        return len(self.imgs)

# train_dataset = DogCat(opt.train_data_root, mode='train')
# trainloader = data.DataLoader(train_dataset, batch_size=opt.batch_size,
#                               shuffle=True,
#                               num_workers=opt.num_workers)

# for ii, (data, label) in enumerate(trainloader):
#     train()