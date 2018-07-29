# Imports here
import argparse
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

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    cat_to_name = get_label_mapping(args.category_names)
    image_path = args.input
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    probs, classes = predict(image_path, model, args.top_k)
    class_names = []
    for i in classes:
        a = inv_dic[i]
        class_names.append(cat_to_name[str(a)])
    print(class_names)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, default=None, help='input image file')
    parser.add_argument('checkpoint', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='return top k most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='json map of names')
    parser.add_argument('--gpu', action='store_true', help='use gpu')

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    if checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    new_classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer'][0])),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(checkpoint['hidden_layer'][0], checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = new_classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def get_label_mapping(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def process_image(image):
    img = Image.open(image)

    img.thumbnail((256, 256))
    img = img.crop((16, 16, 240, 240))

    np_image = np.array(img)/255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    img = (np_image - means)/stds
    tranposed_img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(tranposed_img)

    return tensor

def predict(image_path, model, topk=5):
    img = process_image(image_path)
    model.eval()
    model.to('cpu')
    img.unsqueeze_(0)
    output = model.forward(img.float())
    ps = F.softmax(output, dim=1)

    probs, classes = ps.topk(topk)
    probs = probs.detach().numpy()[0]
    classes = classes.numpy()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes]
    return probs, top_classes

if __name__ == "__main__":
    main()
    
# python predict.py /home/workspace/aipnd-project/flowers/test/1/image_06752.jpg 'checkpoints/checkpoint.pth' --gpu