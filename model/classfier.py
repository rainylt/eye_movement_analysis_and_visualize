import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.input_dim =30
        self.embed = 30
        self.dropout = 0.5
        self.num_classes = 2
        self.num_layers = 2
        self.hidden_size = 128
        self.learning_rate = 1e-3
        self.batch_size = 10
        self.num_epochs = 100
        self.save_path = 'output/save_dict/saved_model.ckpt'
        self.log_path = 'output/log'

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Linear(config.input_dim, config.embed)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size*2, config.num_classes)
        self.init_weights()

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1,:])#最后时刻的hidden state
        out = F.log_softmax(out, dim=1)
        return out

    def init_weights(self):
        for name, w in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w,0)
            else:
                pass
