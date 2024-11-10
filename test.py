import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

'''
    Function to get the device (CPU or GPU) for training
    Use MPS backend if available, otherwise use CUDA if available, otherwise use CPU
'''
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device


'''
    Function to initialize the model based on the model name
'''
def initialize_model(model_name, num_classes, use_pretrained=True):
    model = None
    input_size = 224  
    weights = None

    if model_name == 'resnet':
        """ ResNet18 """
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)


    elif model_name == 'squeezenet':
        """ SqueezeNet """
        weights = models.SqueezeNet1_0_Weights.DEFAULT if use_pretrained else None
        model = models.squeezenet1_0(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes

    elif model_name == 'mobilenet':
        """ MobileNetV2 """
        weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)


    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size

'''
    Function to get the data transforms
'''
def get_data_transforms(input_size):
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

'''
    Function to evaluate the model
'''
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

'''
    Function to plot the confusion matrix
'''
def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def plot_training_curves(histories, model_names):
    for name in model_names:
        history = histories[name]
        train_acc = history['train_acc']
        val_acc = history['val_acc']
        epochs = range(1, len(train_acc) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy for {name.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{name}_accuracy_curve.png')
        plt.close()

def main():
    device = get_device()

    data_dir = './caltech256_subset' 

    val_dataset = datasets.ImageFolder(data_dir)
    class_names = val_dataset.classes
    num_classes = len(class_names)

    input_size = 224  
    data_transforms = get_data_transforms(input_size)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    with open('training_histories.pkl', 'rb') as f:
        histories = pickle.load(f)
    print("Available histories keys:", histories.keys())

    model_names = list(histories.keys())

    for name in model_names:
        print(f'\nEvaluating {name.upper()}...')

        model, _ = initialize_model(name, num_classes, use_pretrained=False)
        model.load_state_dict(torch.load(f'{name}_caltech256.pth', map_location=device))
        model = model.to(device)

        y_true, y_pred = evaluate_model(model, val_loader, device, class_names)

        plot_confusion_matrix(y_true, y_pred, class_names, name)

        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        print(f'Classification Report for {name.upper()}:\n')
        print(report)

        with open(f'{name}_classification_report.txt', 'w') as f:
            f.write(report)

    plot_training_curves(histories, model_names)
    print('Evaluation completed. All graphs and reports have been saved.')


if __name__ == '__main__':
    main()
