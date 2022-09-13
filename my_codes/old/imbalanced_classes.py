import enum
import os

import torch
import torchvision.datasets as datasets
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms


def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    class_weights = []

    for _, _, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1 / len(files))

    sample_weights = [0] * len(dataset)

    for idx, (_, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader


if __name__ == "__main__":
    loader = get_loader("imbalanced", 8)

    num_retrievers = 0
    num_elkhounds = 0

    for epoch in range(15):
        for data, labels in loader:
            num_retrievers += torch.sum(labels == 0)
            num_elkhounds += torch.sum(labels == 1)

    print(num_retrievers, num_elkhounds)
