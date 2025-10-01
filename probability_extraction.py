import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv

# Mean and Std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = "data"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True, num_workers=1)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.show()


def plot(val_loss, train_loss, typ):
    plt.title("{} after epoch: {}".format(typ, len(train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel(typ)
    plt.plot(list(range(len(train_loss))), train_loss, color="r", label="Train " + typ)
    plt.plot(list(range(len(val_loss))), val_loss, color="b", label="Validation " + typ)
    plt.legend()
    plt.savefig(os.path.join(data_dir, typ + ".png"))
    plt.close()


val_loss_gph = []
train_loss_gph = []
val_acc_gph = []
train_acc_gph = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=2, model_name="kaggle"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                print(f"Processing batch {i + 1}/{len(dataloaders[phase])}")

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_gph.append(epoch_loss)
                train_acc_gph.append(epoch_acc)
            if phase == 'val':
                val_loss_gph.append(epoch_loss)
                val_acc_gph.append(epoch_acc)

            plot(val_loss_gph, train_loss_gph, "Loss")
            plot(val_acc_gph, train_acc_gph, "Accuracy")

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(data_dir, model_name + ".pth"))
                print('==> Model Saved')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    model = models.vgg11(pretrained=True)

    num_ftrs = model.classifier[6].in_features  # for vgg11

    print("Number of features: " + str(num_ftrs))

    # Here the size of each output sample is set to num_classes.
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=2, model_name="vgg11")

    # Getting Proba distribution
    print("\nGetting the Probability Distribution")
    testloader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1)

    model.eval()
    correct = 0
    total = 0

    with open(os.path.join(data_dir, "vgg11.csv"), 'w+', newline='') as f:
        writer = csv.writer(f)
        with torch.no_grad():
            num = 0
            temp_array = np.zeros((len(testloader), num_classes))
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                prob = torch.nn.functional.softmax(outputs, dim=1)
                temp_array[num] = prob.cpu().numpy()
                num += 1

            print("Accuracy = ", 100 * correct / total)

            for i in range(len(testloader)):
                writer.writerow(temp_array[i].tolist())

if __name__ == '__main__':
    main()
