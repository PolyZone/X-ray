from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import datasets,transforms
import torchvision.transforms as transforms

from models import Net

def resetSeed(seed=69):
    torch.manual_seed(69)
    torch.cuda.manual_seed_all(69)
    np.random.seed(69)
    random.seed(69)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




net = Net()

path_train = r'C:\Users\Søren\OneDrive - Danmarks Tekniske Universitet\DTU\Bachelor\4. semester\Project work - Bachelor of Artificial Intelligence and Data\Mura_data\MURA-v1.1\train'
path_test = r'C:\Users\Søren\OneDrive - Danmarks Tekniske Universitet\DTU\Bachelor\4. semester\Project work - Bachelor of Artificial Intelligence and Data\Mura_data\MURA-v1.1\valid'

transform = transforms.Compose([transforms.Resize((224, 224)),
                                #transforms.RandomResizedCrop(128),
            transforms.ToTensor()])

master_train = datasets.ImageFolder(path_train, transform = transform)

dataset_train, dataset_valid = torch.utils.data.random_split(master_train, [int(np.floor(0.7*len(master_train))), int(np.ceil(0.3*len(master_train)))])

print(ConcatDataset([dataset_train,dataset_valid]))
print(len(dataset_valid))


dataset = ConcatDataset([dataset_train, dataset_valid])

criterion = nn.CrossEntropyLoss()
num_epochs=20
batch_size = 32
k=5
splits=KFold(n_splits=k,shuffle=True,random_state=69)
foldperf={}


def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        #loss = loss_fn(output, labels)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss, train_correct


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        #loss = loss_fn(output, labels)
        loss = criterion(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct



l2_param = [0, 0.1, 0.25, 0.5]
lr_param = [0.0001, 0.001, 0.01]


count = 0
alt_liste = []

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #     model = Net()
    #     model.to(device)

    for i in lr_param:
        for j in l2_param:
            print(i, j)
            model = Net()
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=i, weight_decay=j)

            history = {'train_loss': [], 'validation_loss': [], 'train_acc': [], 'validation_acc': []}

            for epoch in tqdm(range(num_epochs)):
                train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
                test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_correct / len(train_loader.sampler) * 100
                test_loss = test_loss / len(test_loader.sampler)
                test_acc = test_correct / len(test_loader.sampler) * 100

                print(
                    "Epoch:{}/{} AVG Training Loss:{:.3f} AVG validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG validation Acc {:.2f} %".format(
                        epoch + 1,
                        num_epochs,
                        train_loss,
                        test_loss,
                        train_acc,
                        test_acc))
                history['train_loss'].append(train_loss)
                history['validation_loss'].append(test_loss)
                history['train_acc'].append(train_acc)
                history['validation_acc'].append(test_acc)

                alt = {'fold': [], 'epoch': [], 'L2': [], 'Learning rate': [], 'train_loss': [], 'validation_loss': [],
                       'train_acc': [], 'validation_acc': []}
                alt['fold'].append(fold)
                alt['epoch'].append(epoch)
                alt['L2'].append(j)
                alt['Learning rate'].append(i)
                alt['train_loss'].append(train_loss)
                alt['validation_loss'].append(test_loss)
                alt['train_acc'].append(train_acc)
                alt['validation_acc'].append(test_acc)
                alt_liste.append(alt)
                print(alt_liste[count])
                torch.save(net.state_dict(), r"Model+fold=%s+epoch=%s+lr=%s+L2=%s" % (fold, epoch, i, j))

                count = count + 1

            foldperf['fold{}'.format(fold + 1)] = history







