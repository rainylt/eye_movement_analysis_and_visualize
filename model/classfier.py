import numpy as np
import torch
import torch.nn as nn

class Config(object):
    def __init__(self):
        self.feature_dim =
        self.embed = 300
        self.dropout = 0.5
        self.num_classes = 2
        self.num_layers = 2
        self.hidden_size = 128
        self.learning_rate = 1e-3
        self.batch_size = 10
        
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Linear(config.feature_dim, config.embed)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size*2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1,:])#最后时刻的hidden state
        return out
