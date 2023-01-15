import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" #change number of gpu

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import timm
from PIL import Image

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model,testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            #images, labels = data
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    return correct / total


def train_model(model, model_name, best_acc, dataloaders, criterion, optimizer, transform_type, count=7, early_stopping=24, num_epochs=25):

    history = {}
    history['val_acc'] = []
    history['train_acc'] = []
    best_acc = 0.0
    init_count = count
    init_early_stopping = early_stopping

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            lr = optimizer.param_groups[0]['lr']

            if phase == 'eval':
                print("val_acc: ", epoch_acc.item())
                history['val_acc'].append(epoch_acc)
                if epoch_acc > best_acc:
                    early_stopping = init_early_stopping
                    best_acc = epoch_acc
                    torch.save(model, '5_classes_results/' + model_name + '/' + transform_type + '_' + '_' + "best.pth")
                else: #early stopping
                    count -= 1
                    early_stopping -= 1
                    if count == 0:
                        optimizer.param_groups[0]['lr'] *= 0.1 #if count reaches 0 update the learning rate
                        lr = optimizer.param_groups[0]['lr']
                        count = init_count
            else:
                print("train_acc: ", epoch_acc.item())
                history['train_acc'].append(epoch_acc)

        print(f'model name: {model_name}, learning rate: {lr}, early_stopping: {early_stopping}')
        print('-' * 10)

        if early_stopping == 0:
            break

    return model, history, best_acc

def initialize_model(model_name, num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
    #   Resnet50
        model_ft = timm.create_model('resnet50', pretrained=True, num_classes=num_classes) # change when classifying GAN also 
        input_size = 224

    elif model_name == "mobilenet":
    #   MobileNetv2
        model_ft = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=num_classes)
        input_size = 224

    elif model_name == "vgg16":
    #   Vgg16
        model_ft = models.vgg16(pretrained=True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "xception":
    #   Xception
        model_ft = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        input_size = 224

    elif model_name == "vit":
    #   ViT
        model_ft = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

def save_all(model_name, history, transform_type):
    
    path_train = '5_classes_results/' + model_name + '/train_' + transform_type + '_' + '.txt'
    path_eval = '5_classes_results/' + model_name + '/eval_' + transform_type + '_' + '.txt'
    path_test = '5_classes_results/' + model_name + '/test_' + transform_type + '_' + '.txt'
    for i, path in enumerate([path_train,path_eval,path_test]):
        with open(path, 'w') as f:
            if i == 0:
                ohist = [h.cpu().numpy() for h in history['train_acc']]
                for el in ohist:
                    f.write(str(el)+'\n')
            elif i == 1:
                ohist = [h.cpu().numpy() for h in history['val_acc']]
                for el in ohist:
                    f.write(str(el)+'\n')
            else:
                el = history['test_acc']
                f.write(str(el)+'\n')
            
def create_dir(path):
    if not os.path.exists('5_classes_results/' + path):
        os.mkdir('5_classes_results/' + path)
        print(f'created folder: 5_classes_results/{path}')
    else:
        print(f'directory 5_classes_results/{path} already exists')

def get_data_transform(transform_type, input_size):
    if transform_type == 0:
        transform = transforms.Resize((input_size, input_size))
    elif transform_type == 1:
        transform = transforms.CenterCrop((input_size, input_size))
    elif transform_type == 2:
        transform = transforms.RandomCrop((input_size, input_size))
    else:
        transform = transforms.RandomResizedCrop((input_size,input_size))

    data_transforms = {
    'train': transforms.Compose([
        transform,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'eval': transforms.Compose([
        transform,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transform,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    return(data_transforms)

def main():

    #parameters
    data_dir = 'datasets/5_classes'
    input_size = 224
    workers = 6
    # Models to choose from [resnet, mobilenet, xception, vit]
    model_names = ['vit', 'vgg16', 'mobilenet', 'xception', 'resnet']
    # Batch size for training (change depending on how much memory you have)
    batch_size = 32
    # Number of epochs to train for
    num_epochs = 200
    # number of classes for classification
    num_classes = 5
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create folder to save the results
    create_dir('5_classes_results')

    transform_type = ['resize_only', 'center_crop', 'random_crop', 'random_resized_crop']

    for type, t_type in enumerate(transform_type):
        print(f'starting {type} type.')
        data_transforms = get_data_transform(type, input_size)
        #print(data_transforms)
        print(f'starting {data_dir} directory.')
        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'eval','test']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=workers) for x in ['train', 'eval','test']}
        #loop for all the classifiers
        for model_name in model_names:
            print(f'starting {model_name} classificator.')

            best_acc = 0.0

            create_dir(f'5_classes_results/{model_name}')# + '_' + t_type + '_' + compression_type) # create just one folder for each model
            # Initialize the model for this run
            model_ft, input_size = initialize_model(model_name, num_classes)

            # Send the model to GPU
            model_ft = model_ft.to(device)

            # Observe that all parameters are being optimized
            params_to_update = model_ft.parameters()
            optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9) #we can use Adam

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            model_ft, hist, best_acc = train_model(model_ft, model_name, best_acc, dataloaders_dict, criterion, optimizer_ft, t_type, num_epochs=num_epochs)

            #evaluate the model on the test set
            hist["test_acc"] = evaluate(model_ft,dataloaders_dict['test'])
            

            save_all(model_name, hist, t_type)

	    

if __name__ == '__main__':
    main()
