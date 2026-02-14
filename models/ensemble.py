import torch
import torch.nn as nn

class SoftVotingEnsemble(nn.Module):
    def __init__(self,models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self,x):
        probs_sum = None

        for model in self.models:
            out = torch.softmax(model(x),dim=1)

            if probs_sum is None:
                probs_sum = out
            else:
                probs_sum += out
        
        probs_avg = probs_sum/len(self.models)
        return probs_avg
    
class WeightedEnsemble(nn.Module):
    def __init__(self, models, weights):
        super().__init__()

        assert len(models) == len(weights)

        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights,dtype=torch.float32)

    def forward(self,x):
        probs_sum = None

        for i, model in enumerate(self.models):
            out = torch.softmax(model(x),dim=1) * self.weights[i]

            if probs_sum is None:
                probs_sum = out
            else: 
                probs_sum += out

        probs_avg = probs_sum/self.weights.sum()
        return probs_avg