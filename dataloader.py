import torch
import numpy as np
from torchvision import datasets, transforms
import random

def get_data(master=False):
    path_train = r'C:\Users\sebas\Documents\DTU\6_semester\02466 Project work - Bachelor of Artificial Intelligence and Data\data\MURA-v1.1\train'
    path_test = r'C:\Users\sebas\Documents\DTU\6_semester\02466 Project work - Bachelor of Artificial Intelligence and Data\data\MURA-v1.1\valid'

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    # transforms.RandomResizedCrop(128),
                                    transforms.ToTensor()])

    master_train = datasets.ImageFolder(path_train, transform=transform)

    dataset_train, dataset_valid = torch.utils.data.random_split(master_train, [int(np.floor(0.7 * len(master_train))),
                                                                                int(np.ceil(0.3 * len(master_train)))])
    dataset_test = datasets.ImageFolder(path_test, transform=transform)

    train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
    valid = torch.utils.data.DataLoader(dataset_valid, batch_size=32, shuffle=True)
    test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)

    if master == True:
        return master_train, train, valid, test

    return train, valid, test

def resetSeed(seed=69):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)