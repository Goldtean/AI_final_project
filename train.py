# Imports here
import argparse
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import time
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

import collections
import helper
from PIL import Image

def main():
    args = get_args()
    data_dir = args.directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    image_sets, data_loaders = load_data(data_dir)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
    if args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, args.hidden_units)),
                              ('relu', nn.ReLU()),
                              ('drop', nn.Dropout(p=0.5)),
                              ('fc2', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    if args.arch == 'resnet18':
        model.fc.classifier = classifier
        optimizer = optim.Adam(model.fc.classifier.parameters(), lr=args.learning_rate)
    else:
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
        
    hidden_layers = [args.hidden_units, args.hidden_units/2]
    criterion = nn.NLLLoss()
    train_test_save(model, args.epochs, optimizer, criterion, args.gpu, image_sets, data_loaders, hidden_layers, args.save_dir, input_size, args.arch)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, default = "/flowers", help="data_directory")
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture model vgg16, resnet18, or densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
    
    return parser.parse_args()
    
def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_sets = {}
    image_sets['train_data'] = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    image_sets['valid_data'] = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    image_sets['test_data'] = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    data_loaders = {}
    data_loaders['trainloader'] = torch.utils.data.DataLoader(image_sets['train_data'], batch_size=64, shuffle=True)
    data_loaders['validloader'] = torch.utils.data.DataLoader(image_sets['valid_data'], batch_size=32, shuffle=True)
    data_loaders['testloader'] = torch.utils.data.DataLoader(image_sets['test_data'], batch_size=32)

    return image_sets, data_loaders

def train_test_save(model, epochs, optimizer, criterion, gpu, image_sets, data_loaders, hidden_layers, save_dir, input_size, arch):
    print_every = 25
    steps = 0

    if gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.to(device)
    print(len(data_loaders['trainloader']))
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (images, labels) in enumerate(data_loaders['trainloader']):
            print(steps)
            steps += 1
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs), "Loss: {:.4f}".format(running_loss/print_every))
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in data_loaders['validloader']:
                        images = images.to('cuda')
                        labels = labels.to('cuda')
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()
                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                print("Validation Loss: {:.4f}".format(valid_loss/len(data_loaders['validloader'])),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(data_loaders['validloader'])))
            
            running_loss = 0

    print('Training complete')
    # Set model to evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    model.to('cpu')
    with torch.no_grad():
    
        for data in data_loaders['testloader']:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    model.class_to_idx = image_sets['train_data'].class_to_idx

    checkpoint = {
        'input_size': input_size,
        'output_size': 102,
        'class_to_idx': model.class_to_idx,
        'hidden_layer': hidden_layers,
        'state_dict': model.state_dict(),
        'arch': arch
    }

    torch.save(checkpoint, save_dir+'checkpoint2.pth')

if __name__ == "__main__":
    main()

    # 80% accuracy
    # python train.py '/home/workspace/aipnd-project/flowers' --arch 'densenet121' --gpu --hidden_units 1048 --epochs 2