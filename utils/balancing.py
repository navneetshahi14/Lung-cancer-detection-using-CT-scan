import numpy as np
from collections import Counter;
from sklearn.metrics.pairwise import _num_samples
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


def compute_class_weights(dataset):
    if hasattr(dataset, "indices"):  # Subset case
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:  # ImageFolder case
        labels = dataset.targets
    counts = Counter(labels)

    num_classes = len(counts)
    total = sum(counts.values())

    weights = []
    for i in range(num_classes):
        weights.append(total/ (num_classes * counts[i]))
        # w = np.log(total / (counts[i] + 1e-6)) 
        # weights.append(w)

    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes 
    
    print(weights)

    return torch.tensor(weights,dtype=torch.float32)


def compute_single_class_weights(dataset):
    if hasattr(dataset, "indices"):  # Subset case
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:  # ImageFolder case
        labels = dataset.targets
    counts = Counter(labels)

    num_classes = len(counts)
    total = sum(counts.values())

    weights = []
    for i in range(num_classes):
        # weights.append(total/ (num_classes * counts[i]))
        # w = np.log(total / (counts[i] + 1e-6)) 
        # w = total / counts[i]   # NO log
        w = (total / counts[i]) ** 0.5
        weights.append(w)

    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes 
    
    print(weights)

    return torch.tensor(weights,dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self,inputs,targets):
        ce_loss = F.cross_entropy(inputs,targets,weight=self.alpha,reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
def get_weighted_sampler(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        labels = dataset.targets
        
    class_counts = Counter(labels)
    num_samples = len(labels)
    
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True
    )
    
    return sampler 