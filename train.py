import argparse
import torch
import sys
from torch.autograd import Variable
from network_prep import create_loaders, prep_model, create_Classifier


# Create a dataset on input argument on command line.
data_dir = str(sys.argv[1])
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"

def get_input_args():
    
    '''
    Retrieves and parses command line arguments. This function returns these args
    as an ArgumentParser object.
        7 command line arguments are created:
            data_dir - Nonoptional, path to image files
            save_dir - Path to where checkpoint is stored(default- current path)
            arch - CNN model architecture to use for image classification(default- vgg
                   pick any of the following vgg, densenet, alexnet)
            learning_rate - Learning rate for the CNN(default - 0.001)
            hidden_units - sizes for hidden layers, expects comma seperated if more than one
            output_size - output size for data set training on(default-102)
            epochs - defines number of epochs to run(default- 10)
            gpu - turns gpu training on if selected(default- False/off)
    '''
    parser = argparse.ArgumentParser(description = 'Get NN arguments')
    # Defining arguments 
    parser.add_argument('data_dir', type = str, help = 'mandatory data directory')
    parser.add_argument('--save_dir', default = '', help ='Directory to save checkpoint.')
    parser.add_argument('--arch', default = 'vgg', help = 'default architecture, options: vgg, densenet, alexnet')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='default learning rate' )
    parser.add_argument('--hidden_units', default='512', type=str, help='default hidden layer sizes')
    parser.add_argument('--output_size', default=102, type=int, help='default output_size')
    parser.add_argument('--epochs', default=10, type=int, help='default training epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='use GPU processing')
    
    return parser.parse_args()

def train_flower_Classifier(model, trainloader, validloader, criterion, optimizer, epochs, gpu):
    '''
    Trains the selected model based on parameters passed through from command line arguments. 
    This function performs a validation loop every 40 steps, prints progress of the task and 
    returns the trained model. 
    Parameters: 
        model - CNN architecture to be trained
        trainloader - PyTorch training data
        validloader - PyTorch data to be used for validation
        criterion - loss function to be executed
        optimizer - optimizer function to apply the gradients (default - Adam optimizer)
        epochs - number of epochs to tain on 
        gpu - boolean flags to indicate whether GPU or CPU is being used
    Returns:
        model - A trained CNN model 
    '''
    
    steps = 0
    print_every = 40  # A validation loop of 40 steps
    run_loss = 0
    
    # Selects CUDA processing if gpu is True and we have an environment that supports CUDA
    if gpu and torch.cuda.is_available():
        print("GPU TRAINING")
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print("GPU processing is selected but no NVIDIA drivers found... Training under CPU.")
    else:
        print("CPU TESTING")
    
    for e in range(epochs): 
        print("Starting to train network...")
        print("Training run {}...".format((e + 1)))
        print("\n")
        
        model.train()
        
        # Training forward pass and backpropagation 
        for images, labels in iter(trainloader): 
            steps += 1
            images, labels = Variable(images), Variable(labels)
            if gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
                
                # Clear the gradients as the optimizer gets accumulated with each run
                optimizer.zero_grad()
                
                # Forward-pass, backpropagation and then update the weights
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                run_loss += loss.data.item()
                
                # Runs validation forward pass and loop at specified interval
                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e+1, epochs), "Loss: {:.4f}".format(run_loss/print_every))
                    
                    model.eval()
    
                    accuracy = 0
                    valid_loss = 0
                    
                    for images, labels in iter(validloader):
                        images, labels = Variable(images), Variable(labels)
                    if gpu and torch.cuda.is_available():
                        images, labels = images.cuda(), labels.cuda()
                    # Temporarily turn off gradients (i.e. set requires_grad == False) to preserve memory    
                    with torch.no_grad():
                        out = model.forward(images)
                        valid_loss += criterion(out, labels).data.item()
        
                        # Calculate class probabilities and accuracy
                        # Model's output is log-softmax. Take the exponential to obtain the probabilities
                        ps = torch.exp(out).data
                        # Comparing with true label, class with the highest probability is the predicted class 
                        equality = (labels.data == ps.max(1)[1])
                        # Taking the mean here because accuracy is the number of correct predictions
                        # divided by all predictions
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
#                        accuracy += torch.mean(equality.type(torch.FloatTensor)
        
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(run_loss/print_every),
                          "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                          "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))  
    
                    run_loss = 0
                    model.train()
            
    print("{} EPOCHS COMPLETED. MODEL IS TRAINED.".format(epochs))
    return model 
        
                
def test_flower_Classifier(model, testloader, criterion, gpu):
    
    '''
    Tests the previously trained CNN flower classifier on a test dataset and prints out results.
    Returns nothing. 
    Parameters:
        model - Trained CNN model to test on
        testloader - PyTorch data loader of test data
        criterion - loss function to be executed
        gpu - boolean flag to indicate whether GPU is un use
    Returns:
        None
    '''
    
    # Selecting CUDA if gpu == True and environment supports CUDA
    if gpu and torch.cuda.is_available():
        print("GPU TESTING")
        model.cuda()
    elif gpu and torch.cuda.is_available() == False:
        print("GPU processing is selected but no NVIDIA drivers found... testing under CPU.")
    else:
        print("CPU TESTING")
              
    model.eval()
    
    accuracy = 0 
    run_loss = 0 
    test_loss = 0 
    
    # Forward pass
    for images, labels in iter(testloader):
        images, labels = Variable(images), Variable(labels)
        if gpu and torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            output = model.forward(images)
            test_loss += criterion(output, labels).data.item()
            
            # Calculate class probabilities and the accuracy
            # Model's output is log-softmax, take expoential to obtain the probabilities 
            ps = torch.exp(output).data
            # Comparing with true label, class with the highest probability is the predicted class
            equality = (labels.data == ps.max(1)[1])
            # Taking the mean here because accuracy is the number of correct predictions
            # divided by all predictions
            accuracy += equality.type_as(torch.FloatTensor()).mean()
            
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
           "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
             
                    
def save_model_checkpoint(model, input_size, epochs, save_dir, arch, learning_rate, class_idx, optimizer, output_size):
    '''
    '''
    saved_model = {
    'input_size':input_size,
    'epochs':epochs,
    'arch':arch,
    'hidden_units':[each.out_features for each in model.classifier if hasattr(each, 'out_features') == True],
    'output_size': output_size,
    'learning_rate': learning_rate,
    'class_to_idx': class_idx,
    'optimizer_dict': optimizer.state_dict(),
    'classifier': model.classifier,
    'state_dict': model.state_dict() 
    }
    #Save checkpoint in current directory unless otherwise specified by save_dir
    if len(save_dir) == 0:
        save_path = save_dir + 'checkpoint.pth'
    else:
        save_path = save_dir + '/checkpoint.pth'
    torch.save(saved_model, save_path)
    print('Model saved at {}'.format(save_path))
    
    
def main(): 
    in_args = get_input_args()
    trainloader, testloader, validloader, class_idx = create_loaders(in_args.data_dir)
    model, input_size = prep_model(in_args.arch)
    model, criterion, optimizer = create_Classifier(model, input_size, in_args.hidden_units, in_args.output_size,
                                                    in_args.learning_rate)
    trained_model = train_flower_Classifier(model, trainloader, validloader, criterion, optimizer, in_args.epochs,
                                     in_args.gpu)
    test_flower_Classifier(trained_model, testloader, criterion, in_args.gpu)
    save_model_checkpoint(trained_model, input_size, in_args.epochs, in_args.save_dir, in_args.arch,
                          in_args.learning_rate, class_idx, optimizer, in_args.output_size)
     
if __name__ == '__main__':
    main()
