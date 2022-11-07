import torch
import argparse
import os
import time
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json

from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image
from datasets import ImageFolder
from torch.utils.data import DataLoader
# from train.py import process_the_image
def read_categories():
    if (args.category_names is not None):
        cat_file = args.category_names 
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None
def dispay_the_prediction(answer):
    cat_files=read_categories()
    i=0
    for a,b in answer:
        i+=1
        p=str(round(a,4)*100)+"%"
        if cat_files:
            b=cat_files.get(str(b),"None")
        else:
            b="class {}".format(str(b))
        print("{}.{} ({})".format(i,b,p))
     return None   
def load_the_model():
    model_detail=torch.load(args.model_checkpoint)
    model=model_detail["model"]
    model.classifier=model_detail["classifier"]
    model.load_state_dict(model_detail["state_dict"])
    return model
def process_the_image(image):
    im=Image.open(image)
    width,height=im.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if (max_element == 0):
        min_element = 1
    else:
        min_element = 0
    aspect_ratio=picture_coords[max_element]/picture_coords[min_element]
    new_picture_coords = [0,0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)   
    width, height = new_picture_coords
    left = (width - 244)/2
    top = (height - 244)/2
    right = (width + 244)/2
    bottom = (height + 244)/2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image = np_image / [255,255,255]
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def classify_the_image(image_path,topk=5):
    topk=int(topk)
    with torch.no_grad():
        image=process_the_image(image_path)
        image=torch.from_numpy(image)
        image.unsqueeze_(0)
        image=image.float()
        model=load_the_model()
        if args.gpu:
            image=image.cuda()
            model=model.cuda()
        else:
            image=image.cpu()
            model=model.cpu()
        output=model(image)
        prob,classes=torch.exp(outputs).topk(topk)
        prob,classes=prob[0].tolist(),classes[0].add(1).tolist()
        results=zip(prob,classes)
        return results
def parse():
    parser = argparse.ArgumentParser(description='use a neural network to classify an image!')
    parser.add_argument('image_input', help='image file to classifiy (required)')
    parser.add_argument('model_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    return args
def main():
    global args
    args = parse() 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU is detected")
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
    image_path = args.image_input
    predict = classify_the_image(image_path,top_k)
    display_the_prediction(predict)
    return predict

main()