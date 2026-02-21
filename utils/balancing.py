import numpy as np
from collections import Counter;
import torch

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

    return torch.tensor(weights,dtype=torch.float32)

