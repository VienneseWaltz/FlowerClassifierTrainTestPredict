import json
import numpy as np                                      
import sys                                     
import torch
import torch.nn.functional as F   

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict

data_dir = str(sys.argv[1])
#data_dir = 'flowers'
def create_loaders(data_dir):    
    '''
    Creates datasets - PyTorch training, testing and validation dataloaders
    Input Parameters: 
        data_dir - Path to data to be used
    Returns:
        trainloader - Normalized training data
        testloader - Normalized testing data
        validloader - Normalized validation data 
     '''
    
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    
    # Converting images to PyTorch tensors
    # Defining data transforms for valid datasets 
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_std)])
    
    # Defining train transforms
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(norm_mean, norm_std)])
    
    # Defining test transforms
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, 
                                                               norm_std)])
                                    
    
    # *** LOADING DATASETS ***
    # Pass transforms in here with ImageFolder, then run the next cell and see how the transforms look
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = data_transforms) 
    
    # Defining dataloaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle = True)
    validloader = DataLoader(valid_dataset, batch_size=32, shuffle = True)

    # Attach train_data.class_to_idx to class_idx as an attribute to make it
    # easier for making inference later on. 
    class_idx = train_dataset.class_to_idx
    
    return trainloader, testloader, validloader, class_idx

def prep_model(arch): 
    
    '''
    Selects, downloads and returns the model architecture for the CNN
    and provides the associated input size for each. 
    
    Input Parameters :
        arch - Used to select the architecutre to use
    Returns:
        model_select[arch] - selects variable out of a dictionary and returns
                             the model associated with arch
        input_size[arch] - selects the associated input size for the model selected
                           with arch                      
    '''
    
    alexnet = ''
    densenet121 = ''
    vgg16 = ''
    if arch == 'alexnet':
        alexnet = models.alexnet(pretrained = True)
    elif arch == 'densenet':
        densenet121 = models.densenet121(pretrained = True)
    elif arch == 'vgg':
        vgg16 = models.vgg16(pretrained = True)
    else:
        print("Architecture not recognized.")
        sys.exit()
        
   # Process the model (which is a dictionary) selected, and obtain the associated input size for each CNN model
    model_selected = {'alexnet':alexnet, 'densenet':densenet121, 'vgg':vgg16}
    input_size = {'alexnet':9216, 'densenet':1024, 'vgg':25088}
    return model_selected[arch], input_size[arch] 
   
def create_Classifier(model, input_size, hidden_layers, output_size, learning_rate, drop_p=0.5):
    #Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_layers = hidden_layers.split(',')
    hidden_layers = [int(x) for x in hidden_layers]
    hidden_layers.append(output_size)
    
    # Take hidden_layer sizes and creates layer size definitions for each hidden_layer size combination
    layers = nn.ModuleList([nn.Linear(input_size,hidden_layers[0])])
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])   
    
    # Preserves the order in which the keys of net_layers are inserted
    net_layers = OrderedDict()
    
    # Dropout value
    dropout = 0.33
    
    # Creates hidden layers for each size passed by hidden_layers arg
    for x in range(len(layers)):
        layerid = x + 1
        if x == 0:
            net_layers.update({'drop{}'.format(layerid):nn.Dropout(p=dropout)})
            net_layers.update({'fc{}'.format(layerid):layers[x]})
        else:
            net_layers.update({'relu{}'.format(layerid):nn.ReLU()})
            net_layers.update({'drop{}'.format(layerid):nn.Dropout(p=dropout)})
            net_layers.update({'fc{}'.format(layerid):layers[x]})
        
    net_layers.update({'output':nn.LogSoftmax(dim=1)})
    
    #Define classifier
    classifier = nn.Sequential(net_layers)
    
    #Apply new classifier and generate criterion and optimizer
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def load_model(checkpoint):
    trained_model = torch.load('checkpoint.pth')
    arch = trained_model['arch']
    class_idx = trained_model['class_to_idx']
    #Only download the model you need, kill program if one of the three models isn't passed
    if arch == 'alexnet':
        load_model = models.alexnet(pretrained=True)
    elif arch == 'densenet':
        load_model = models.densenet121(pretrained=True)
    elif arch == 'vgg':
        load_model = models.vgg16(pretrained=True)
    else:
        print('{} architecture not recognized. Supported args: \'vgg\', \'alexnet\', or \'densenet\''.format(arch))
        sys.exit()
        
    for param in load_model.parameters():
        param.requires_grad = False
    
    load_model.classifier = trained_model['classifier']
    load_model.load_state_dict(trained_model['state_dict'])
    
    return load_model, arch, class_idx

if __name__ == '__main__':
    print('This is run as main.')
