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

def parse():
#     this function read the arguments and store the the parameter and input values which are given by commnad line
    parser = argparse.ArgumentParser(description='Train a neural network with open of many options!')
    parser.add_argument('data_directory', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='gpu')
    args = parser.parse_args()
    return args
def processing_the_data(data_dir):
#     processing _the_data function transform the image dataset into dataloader 
    train_dir,test_dir,valid_dir=data_dir
#     Compose is used for data augmentation. we resize the image and do centercrop and then convert it into tensor anf the normalize the image so that pixel values remain between -1 and 1
    transform=transforms.Compose([ transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])  
    train_dataset=ImageFolder(train_dir,transform=transform)
    valid_dataset=ImageFolder(valid_dir,transform=transform)
    test_dataset=ImageFolder(test_dir,transform=transform)
    train_dl=DataLoader(train_dataset,batch_size=64,shuffle=True)
    valid_dl=DataLoader(valid_dataset,batch_size=64,shuffle=True)
    test_dl=DataLoader(test_dataset,batch_size=64,shuffle=True)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train':train_dl,'valid':valid_dl,'test':test_dl,'labels':cat_to_name}
#     we have created json file to store valid ,train test dataloader which can be used in further process
    return loaders
def test_the_accuracy(model,dl,device="cpu"):
#     this process is used in test the accuracy. here model is trained model and dl is dataloader and device is eithre cpu or gpu 
    corr=total=0
#     correct  and total output initialize as zero
    with torch.no_grad():
        for image,label in dl:
            image,label=image.to(device),label.to(device)
#             load image and label into device
            output=model(image)
            _,pred=torch.max(output.data,1)
#         find greates probability among all possible output
            total+=label.size(0)
            corr+=(pred==label).sum().item()
#         if pred output is same as correct out then it should add to correct pout put
    return corr/total        
def save_model(model):
#     save_model is used to save the model. this function is used for check point purpose. 
# this function is used to save the model state and trained model so that we can use it in further prediction
    if (args.save_dir is None:
        save_dir="save_check.pth"
    else:
        save_dir=args.save_dir
    checkpoint={'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0    
def validation():
#         this function is used to check whether all input is valid or not
    print("*****validating parameters******")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('directory does not exist!')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('missing: test, train or valid sub-directories')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one of: vgg or densenet')  
def train_model(model,data):
#         train_model is used to train the model
        print("___traing__the model___")
        print_every=10
        learn_rate=args.learning_rate
        if args.learning_rate is None:
            learn_rate=0.003
        epochs=args.epochs
        if args.epochs is None:
             epochs=10
        if args.gpu:
            device="cuda"
        else:
            device="cpu"
        learn_rate=float(learn_rate)
        epochs=int(epochs)
        train_dl=data["train"]
        valid_dl=data["valid"]
        test_dl=data["test"]
        LOSS=nn.NLLoss()
        opt=optim.Adam(model.classifier.parameters(),lr=learn_rate)
        step_no=0
        model.to(device)
        for epoch in range(epochs):
              running_loss=0
        for ii ,(images,labels) in enumerate(train_dl):
            step_no += 1
            
            inputs, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            
            outputs = model.forward(inputs)
            loss = LOSS(outputs, labels)
            loss.backward()
            opt.step()     
            
            running_loss += loss.item()
            
            if step_no % print_every == 0:
                valid_accuracy = test_the_accuracy(model,valid_dl,device)
                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))            
                running_loss = 0
    print("DONE TRAINING!")
    test_result = test_the_accuracy(model,test_dl,device)
    print('final accuracy on test set: {}'.format(test_result))
    return model

def build_new_model(data):
        if (args.arch is None:
            arch_type="vgg"
        else:
            arch_type=args.arch
        if arch_type=="vgg":
            model=torchvision.models.vgg13(pretrained=True)
            input_node=25088
        elif arch_type=="resnet":
            model=torchvision.models.resnet50(pretrained=True)
            input_node=1024
        if (args.hidden_units is None:
            hidden_units=1024
        else:
               hidden_units=int(args.hidden_units)
        for param in model.parameters():
            param.requires_grad=False
        classifer=nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(input_node, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
         model.classifier=classifer
         return model   
            
def get_new_data():
    train_dir=args.data_directory+"/train"
    test_dir=args.data_directory+"/test"
    valid_dir=args.data_directory+"/valid"
    data_dir=[train_dir,test_dir,valid_dir]
    return processing_the_data(data_dir)
def create_new_model():
    validation()
    data=get_new_data()
    model=build_new_model(data)
    model=train_model(model,data)
    
def main():
    global args
    args=parse()
    create_new_model()
    return None
main()