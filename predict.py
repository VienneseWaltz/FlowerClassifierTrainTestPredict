import argparse
import torch
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import json
import numpy as np
from network_prep import load_model


def get_input_args():
    '''
    Function parses command line arguments and returns them as an ArgumentParser object. 
    The result is 5 command line arguments are created:
            input - path to image file to predict
            checkpoint - path to checkpoint file
            top_k - number of predictions to output(default- 1)
            category_names - path to json file for mapped names(default- None)
            gpu - tells the program whether or not to select GPU processing(default - False/off)
        Parameters:
            None - use module to create & store command line arguments
        Returns:
            parse_args() - Storing the command lne arguments as an ArgumentParser object 
    '''
    parser = argparse.ArgumentParser(description='Obtain Neural Network arguments')
    #Define arguments
    parser.add_argument('input', type=str, help='image to process and predict')
    parser.add_argument('checkpoint', type=str, help='cnn to load')
    parser.add_argument('--top_k', default=1, type=int, help='default top_k results')
    parser.add_argument('--category_names', default='', type=str, help='default category file' )
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU processing')
    
    return parser.parse_args()

 
def load_checkpoint(file_dir):
    '''
    A function that loads a checkpoint and rebuilds the model 
    '''
    checkpoint = torch.load(file_dir)
    
    # Get model architecture 
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet(pretrained = True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    else: 
        print("Model architecture could not be found ")
        
    # Features parameters
    for x in model.parameters():
        x.requires_grad = False
        
    # Load model parameters
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 
                                                     
# Processing the image
def process_image(image):
    ''' 
    This function scales, crops and normalizes a PIL image. It then resizes 
    and transposes the image into a new image for PyTorch model. 
    Returns - a Numpy array
    '''
    from PIL import Image
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    # Creates a 256 x 256 thumbnail of the JPEG image
    img.thumbnail((256,256), Image.ANTIALIAS)
    
    # Cropping the image 
    img = img.crop((16, 16, 240, 240))
    
    img = np.array(img)
    
    # Normalize and resize array
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    np_image = img/255
    np_image = (np_image - mean)/std
    
    # PyTorch tensors expect the color channel to be the first dimension but
    # it's the third dimension in PIL image and Numpy array
    new_image = np_image.transpose((2, 0, 1))
    
    # Convert Numpy array to a Tensor array
    new_image = torch.tensor(new_image)
    
    return new_image
    

def predict(image, model, top_k, gpu, category_names, arch, class_idx):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model. Returns top_k classes
    and probabilities. If name json file is passed, it will convert classes to actual names.
    '''
    image = image.unsqueeze(0).float()
    
    image = Variable(image)
    
    if gpu and torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
        print("GPU PROCESSING")
    else:
        print("CPU PROCESSING\n")
    with torch.no_grad():
        output = model.forward(image)
        results = torch.exp(output).data.topk(top_k)
    classes = np.array(results[1][0], dtype=np.int)
    probs = Variable(results[0][0]).data
    
    #If category file path passed, convert classes to actual names
    if len(category_names) > 0:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        #Creates a dictionary of loaded names based on class_ids from model
        mapped_names = {}
        for k in class_idx:
            mapped_names[cat_to_name[k]] = class_idx[k]
        #invert dictionary to accept prediction class output
        mapped_names = {v:k for k,v in mapped_names.items()}
        
        classes = [mapped_names[i] for i in classes]
        probs = list(probs)
    else:
        #Invert class_idx from model to accept prediction output as key search
        class_idx = {v:k for k,v in class_idx.items()}
        classes = [class_idx[i] for i in classes]
        probs = list(probs)
    return classes, probs

def print_predict(classes, probs):
    """
    Prints predictions associated with each class.
    Parameters:
        classes - list of predicted classes
        probs - list of probabilities associated with class from classes with the same index
    Returns:
        None - Use module to print predictions
    """
    predictions = list(zip(classes, probs))
    for i in range(len(predictions)):
        print(' Predicted classes = {}, Probabilities = {:.3%}'.format(predictions[i][0], predictions[i][1]))

def main():
    in_args = get_input_args()
    norm_image = process_image(in_args.input)
    model, arch, class_idx = load_model(in_args.checkpoint)
    classes, probs = predict(norm_image, model, in_args.top_k, in_args.gpu, in_args.category_names, arch, class_idx)
    print_predict(classes, probs)
    
if __name__ == '__main__':
    main()        
