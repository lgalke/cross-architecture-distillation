from torch import nn
import torch
import torchsummary

from src.third_party.seq2mat.seq2mat import Seq2matPreTrainedModel, Seq2matModel
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.third_party.seq2mat.seq2mat import (
    Seq2matForSequenceClassification
)


class LeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LeNet, self).__init__()
        # outout size is same as input
        self.conv1 = nn.Conv1d(in_channels, 6, 17, padding=0)
        self.poo1 = nn.MaxPool1d(4, stride=4)  # output size with hybrid: 198
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.poo2 = nn.AvgPool2d(2, stride=2)
        self.flat = nn.Flatten()
        self.lin = nn.Linear(400, 120)
        self.lin2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, train=True):

        # [batch, in_channels, embedding-size] -> on hybrid [16,1,800]
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.poo1(x)
        x = x.reshape((x.shape[0], x.shape[1], 14, 14))
        x = self.conv2(x)
        x = self.relu(x)
        x = self.poo2(x)
        x = self.flat(x)
        x = self.lin(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.out(x)

        return x


class CNN_Student(Seq2matForSequenceClassification):
    def __init__(self, config):
        super(CNN_Student, self).__init__(config)
        if(config.num_labels < 1):
            config.num_labels = config.num_output_labels

        config.num_output_labels = config.num_labels
        self.num_labels = config.num_labels

        self.classifier = LeNet(
            1,  self.config.num_labels)

    def load_pretrained_embeddings(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        if weights is not None and weights['type'] == self.mode:
            for assignment in weights['assignments']:
                # self.transformer.embeddings.matrix_embedding.embedding
                self.transformer.embeddings.matrix_embedding.embedding.weight[assignment['bert_index']].data.copy_(
                    torch.tensor(assignment['embedding']).view(-1).data)
