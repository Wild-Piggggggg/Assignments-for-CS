from models.resnet import ResNet
from utils.visualize import Visualizer
from config import opt
from data.dataset import DogCat
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torchnet import meter
import torch.nn.functional as F

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import ipdb


# visualizer = Visualizer()
# opt = DefaultConfig()
# dataset = DogCat(opt.train_data_root, mode='train')
# model = ResNet()

@torch.no_grad()
def val(model, val_loader):
    """
    acc on val data
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    with torch.no_grad():
        for batch_idx, (data, label) in tqdm(enumerate(val_loader)):
            data, label = data.to(opt.device), label.to(opt.device)
            output = model(data)
            confusion_matrix.add(output.detach().squeeze(),label.type(torch.LongTensor))
        
    model.train()
    cm_value = confusion_matrix.value()
    print(cm_value)
    accuracy = 100*(cm_value[0][0]+cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer()

    # step1: define the network
    model = ResNet(num_classes=2)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: process data
    train_data = DogCat(opt.train_data_root,mode='train')
    val_data = DogCat(opt.train_data_root,mode='val')
    train_loader =  DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_loader = DataLoader(val_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)

    # step3: define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    lr=opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = opt.weight_decay)

    # step4: metrics 
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # step5: train
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            data, label = data.to(opt.device), label.to(opt.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # update metrics and visualization
            loss_meter.add(loss.item())
            confusion_matrix.add(output.detach(), label.detach())

            if (batch_idx+1)%opt.print_freq == 0:
                step = epoch*len(train_loader)+batch_idx+1
                vis.log_scalar('Loss/train', loss.item(), step)
                # ipdb.set_trace()

        model.save()

        # calculate the metric on val data
        val_cm, val_acc = val(model, val_loader)
        vis.log_scalar('Acc/val', val_acc, epoch)

        if loss_meter.value()[0] > previous_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def write_csv(result ,file_path):
    """
    result: [(path1, prob1), (path2, prob2), ...]
    file_path: the target save path
    """
    df = pd.DataFrame(result,columns=['id', 'prob'])
    df.to_csv(file_path,index=False)
    print(f"Result saved to {file_path}!")

def test(**kwargs):
    opt.parse(kwargs)

    model = ResNet(num_classes=2).eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:model.cuda()

    test_data = DogCat(opt.test_data_root, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    result = []
    for batch_idx, (data, path) in tqdm(enumerate(test_dataloader)):
        data = data.to(opt.device)
        output = model(data)
        probability = F.softmax(output,dim=1).data.tolist()
        batch_results = [(path_, prob_) for path_, prob_ in zip(path.item(), probability)]
        result+=batch_results
    
    write_csv(result, opt.result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the descript")
    parser.add_argument('--mode', choices=['train', 'test'], help='train or test')
    parser.add_argument('--train_data_root', default='./data/train/', help='train_data_root')
    parser.add_argument('--test_data_root', default='./data/test1/', help='test_data_root')
    parser.add_argument('--load_model_path', default=None, help='model path')

    
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(**vars(args))
    
    elif args.mode == 'test':
        test(**vars(args))
