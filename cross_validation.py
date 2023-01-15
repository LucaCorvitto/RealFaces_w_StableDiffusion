import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" #change number of gpu

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

def save_all(model_name, history, src_t_type, src_c_type, dst_t_type, dst_c_type, dir_path):
    path = dir_path + '/' + model_name + '_' + src_t_type + '_' + src_c_type + '---' + dst_t_type + '_' + dst_c_type + '.txt'
    with open(path, 'w') as f:
        el = history['test_acc']
        f.write(str(el)+'\n')
    print('test completed on:', model_name + '_' + src_t_type + '_' + src_c_type + '---' + dst_t_type + '_' + dst_c_type)

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'created folder: {path}')
    else:
        print(f'directory {path} already exists')

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
    hist = {}
    data_dirs = [('datasets/png_images', 'png'),('datasets/jpg100_images', 'jpg100'),
                    ('datasets/jpg90_images', 'jpg90'), ('datasets/jpg70_images', 'jpg70')]
    input_size = 224
    workers = 6
    # Models to choose from [resnet, mobilenet, xception, vit]
    model_names = ['vit', 'vgg16', 'mobilenet', 'xception', 'resnet']
    # Batch size for training (change depending on how much memory you have)
    batch_size = 32
    create_dir('datasets/cross_validation')

    # resize images
    transform_type = ['resize_only', 'center_crop', 'random_crop', 'random_resized_crop']


    for type,dst_t_type in enumerate(transform_type):
            # if dst_t_type != 'resize_only':
            #     continue
        data_transforms = get_data_transform(type, input_size)
        for data_dir,dst_c_type in data_dirs:
            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
            # Create training and validation dataloaders
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=workers) for x in ['test']}
            #loop for all the classifiers
            for model_name in model_names:
                for _, t_type in enumerate(transform_type):
                    # if t_type != 'resize_only':
                    #     continue
                    for _, c_type in data_dirs:
                        if dst_t_type == t_type and dst_c_type == c_type:
                            continue
                        dir_path = 'datasets/cross_validation/' + t_type + '---' + dst_t_type
                        create_dir(dir_path)
                        model_ft = torch.load(model_name + '/' + t_type + '_' + c_type + '_' + "best.pth") #load model of t_type and c_type
                        #evaluate the model on the test set
                        hist["test_acc"] = evaluate(model_ft, dataloaders_dict['test'])
                        save_all(model_name, hist, t_type, c_type, dst_t_type, dst_c_type, dir_path)

if __name__ == '__main__':
    main()
