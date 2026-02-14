import numpy as np
from collections import Counter;
import torch

def compute_class_weights(dataset):
    labels = [label for _,label in dataset.samples]
    counts = Counter(labels)

    num_classes = len(counts)
    total = sum(counts.values())

    weights = []
    for i in range(num_classes):
        weights.append(total/ (num_classes * counts[i]))

    return torch.tensor(weights,dtype=torch.float32)

