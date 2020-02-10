import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models.mlp import FcNet
from utils.merge_bn import merge_bn_in_Sequential
from utils.dataloader import MyDataset

import argparse
import os
# import shutil
import copy

def train(log_interval, model, device, train_loader, optimizer, 
          epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.sampler)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))

def main(model, train_loader, test_loader, device, epochs = 30, parallel = False, lr = 1e-3, 
        lr_decay_interval=10, lr_decay_factor=0.1, log_interval=1):
    
    model.to(device)
    if parallel:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        if (epoch) % lr_decay_interval == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr'] * lr_decay_factor
        train(log_interval, model, device, train_loader, optimizer, 
              epoch)
        test(model, device, test_loader)

    return model



if __name__ == '__main__':
    '''
    python train_model.py --cuda 0 --num_layers 10 --num_neurons 20  --activation relu --dataset mnist --num_epochs 50 --lr 0.1 --lr_decay_interval 10 --lr_decay_factor 0.1
    '''

    parser = argparse.ArgumentParser(description='Train Model Terminal Runner')
    parser.add_argument('--cuda', type=int, default=0, help='gpu idx to use')
    parser.add_argument('--num_layers', type=int, default=7, help='number of layers')
    parser.add_argument('--num_neurons', type=int, default=20, help='number of neurons per layer')
    parser.add_argument('--activation', type=str, default='tanh', help='nonlinear activation')
    parser.add_argument('--dataset', type=str, default='mnist', help='the dataset to use')

    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_interval', type=int, default=10, help='learning rate decay interval')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--log_interval', type=int, default=1, help='log printing interval')

    args = parser.parse_args()

    if args.dataset == 'mnist':
        save_dir = 'pretrained_models/mnist/net_%s_%s_%s/' % (
                    str(args.num_layers), str(args.num_neurons), args.activation)
    elif args.dataset == 'cifar10':
        save_dir = 'pretrained_models/cifar10/net_%s_%s_%s/' % (
                    str(args.num_layers), str(args.num_neurons), args.activation)
    elif args.dataset == 'drive':
        save_dir = 'pretrained_models/drive/net_%s_%s_%s/' % (
                    str(args.num_layers), str(args.num_neurons), args.activation)
    else:
        raise Exception('%s dataset is not supported' % args.dataset)

    save_models = True
    merge_bn = True
    parallel = False

    if args.cuda < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%s' % (str(args.cuda)))
    input_dimension_of_all_datasets = {'mnist':28*28, 'cifar10':3*32*32, 'drive':48}
    input_dimension = input_dimension_of_all_datasets[args.dataset]

    output_dimension_of_all_datasets = {'mnist':10, 'cifar10':10, 'drive':11}
    output_dimension = output_dimension_of_all_datasets[args.dataset]
    # set up model structure
    net = FcNet(num_layers=args.num_layers, num_neurons=args.num_neurons, input_dimension=input_dimension, 
                output_dimension=output_dimension, bn=True, affine=True, activation=args.activation, 
                final_bn = False, dropout=False)
    
    net.to(device)
    print(net)
    
    if args.dataset == 'mnist':
        # set up mnist dataloader
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('datasets/mnist', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])),
                        batch_size=256, shuffle=True) #, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('datasets/mnist', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])),
                            batch_size=500, shuffle=True) #, **kwargs)
    
    elif args.dataset == 'cifar10':
    # set up cifar10 loader
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('datasets/cifar10', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                        ])),
                        batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../datasets/cifar10', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                        ])),
                            batch_size=250, shuffle=True)
    elif args.dataset == 'drive':
    # set up drive loader
        train_loader = torch.utils.data.DataLoader(
            MyDataset('datasets/drive/train_data.ckpt', normalize=True),
                        batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            MyDataset('datasets/drive/test_data.ckpt', normalize=True),
                            batch_size=250, shuffle=True)
    else:
        raise Exception('%s dataset is not supported' % args.dataset)

    net = main(net, train_loader, test_loader, device, epochs = args.num_epochs, parallel = parallel, lr = args.lr, 
            lr_decay_interval=args.lr_decay_interval, lr_decay_factor=args.lr_decay_factor, log_interval=args.log_interval)

    if merge_bn:
        net_no_bn = copy.deepcopy(net.eval())
        net_no_bn.model = merge_bn_in_Sequential(net_no_bn.model)
    if (not save_dir is None) and save_models:
        os.makedirs(save_dir, exist_ok=True)
        # shutil.copyfile(__file__, save_dir+__file__)
        if not parallel:
            torch.save(net.cpu().state_dict(), save_dir+'net')
            print('Saved model to '+save_dir+'net')
            if merge_bn:
                torch.save(net_no_bn.cpu().state_dict(), 
                        save_dir+'merged_bn_net')
            print('Saved model to '+save_dir+'merged_bn_net')
        else:
            torch.save(net.module.cpu().state_dict(), save_dir+'net')
            print('Saved model to '+save_dir+'net')
            if merge_bn:
                torch.save(net_no_bn.module.cpu().state_dict(), 
                        save_dir+'merged_bn_net')
                print('Saved model to '+save_dir+'merged_bn_net')
    