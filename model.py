import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *x.shape[-3:])  # (samples * timesteps, input_size)
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, classes):
        super(RNNClassifier, self).__init__()
        self.time_dist_resnet = TimeDistributed(torchvision.models.resnet50(pretrained=True), batch_first=True)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_size, classes)

    def forward(self, x):
        embeddings = self.time_dist_resnet(x)
        outputs, hidden = self.gru(embeddings)
        return self.linear(outputs[:, -1])
