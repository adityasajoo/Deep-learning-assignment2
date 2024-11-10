import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import time
import copy
import pickle
import os
from torchvision.models import (
    ResNet18_Weights,
    AlexNet_Weights,
    SqueezeNet1_0_Weights,
    MobileNet_V2_Weights,
    ShuffleNet_V2_X1_0_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
)

'''
    Function to get the device (CPU or MPS)
'''
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    return device


'''
    Function to initialize the model based on the model name
'''
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model = None
    input_size = 224  
    weights = None

    if model_name == 'resnet':
        """ ResNet18 """
        if use_pretrained:
            weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)


    elif model_name == 'squeezenet':
        """ SqueezeNet """
        if use_pretrained:
            weights = SqueezeNet1_0_Weights.DEFAULT
        model = models.squeezenet1_0(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes

    elif model_name == 'mobilenet':
        """ MobileNetV2 """
        if use_pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size

'''
    Function to set requires_grad flag to False for feature extracting
'''
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


'''
    Function to get the data transforms
'''
def get_data_transforms(input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  
                                 [0.229, 0.224, 0.225]) 
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

'''
    Main training function
'''
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, scheduler=None):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save history
            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu().item())
            else:
                val_acc_history.append(epoch_acc.cpu().item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history


def main():
    device = get_device()

    data_dir = './caltech256/256_ObjectCategories' 

    num_classes_to_use = 50  
    selected_classes = [str(i).zfill(3) for i in range(1, num_classes_to_use+1)]
    class_dirs = [os.path.join(data_dir, c) for c in os.listdir(data_dir) if c.split('.')[0] in selected_classes]
    data_subset_dir = './caltech256_subset'
    if not os.path.exists(data_subset_dir):
        os.makedirs(data_subset_dir)
        for class_dir in class_dirs:
            os.symlink(os.path.abspath(class_dir), os.path.join(data_subset_dir, os.path.basename(class_dir)))
    data_dir = data_subset_dir

    models_to_train = ['resnet']  
    models_pretrained = ['squeezenet', 'mobilenet']

    models_dict = {}
    histories = {}

    input_size = 224 
    data_transforms = get_data_transforms(input_size)
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    num_classes = len(full_dataset.classes)

    targets = [s[1] for s in full_dataset.samples]
    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset)),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    global dataset_sizes
    dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}
    class_names = full_dataset.classes

    for name in models_to_train:
        print(f'\nTraining {name.upper()} from scratch...')

        model, input_size = initialize_model(name, num_classes, feature_extract=False, use_pretrained=False)
        model = model.to(device)

        data_transforms = get_data_transforms(input_size)

        full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        batch_size = 32  
        num_workers = 4  
        dataloaders = {
            'train': DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers),
            'val': DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
        }

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        num_epochs = 10 

        model, train_hist, val_hist = train_model(model, dataloaders, criterion, optimizer, num_epochs, device, scheduler)

        torch.save(model.state_dict(), f'{name}_caltech256.pth')
        print(f'Model saved as {name}_caltech256.pth')

        models_dict[name] = model
        histories[name] = {'train_acc': train_hist, 'val_acc': val_hist}

    for name in models_pretrained:
        print(f'\nFine-tuning pre-trained {name.upper()}...')

        model, input_size = initialize_model(name, num_classes, feature_extract=True, use_pretrained=True)
        model = model.to(device)

        data_transforms = get_data_transforms(input_size)

        full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        batch_size = 32 
        num_workers = 4 
        dataloaders = {
            'train': DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers),
            'val': DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
        }

        params_to_update = [param for param in model.parameters() if param.requires_grad]

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        num_epochs = 5  

        model, train_hist, val_hist = train_model(model, dataloaders, criterion, optimizer, num_epochs, device)

        torch.save(model.state_dict(), f'{name}_caltech256_pretrained.pth')
        print(f'Model saved as {name}_caltech256_pretrained.pth')

        models_dict[name] = model
        histories[name] = {'train_acc': train_hist, 'val_acc': val_hist}

    with open('training_histories.pkl', 'wb') as f:
        pickle.dump(histories, f)
    print('Training histories saved as training_histories.pkl')

if __name__ == '__main__':
    main()
