import torch
import torch.nn as nn

class SoftVotingEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):

        logits_sum = None

        for model in self.models:
            logits = model(x)

            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

        logits_avg = logits_sum / len(self.models)

        return logits_avg
    
class WeightedEnsemble(nn.Module):
    def __init__(self, models, weights):
        super().__init__()

        assert len(models) == len(weights)

        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, x):

        logits_sum = None
        weights = self.weights.to(x.device)

        for i, model in enumerate(self.models):

            logits = model(x) * weights[i]

            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

        logits_avg = logits_sum / weights.sum()

        return logits_avg