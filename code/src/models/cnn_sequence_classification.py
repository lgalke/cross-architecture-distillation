from src.models.model_template import Seq2MatModelTemplate
from torch import nn
import torch
import math
import pickle


def same_padding(input_size, kernel_size=3, stride=1) -> int:
    """calculates padding so output_size and input_size are equal

    Args:
        input_size ([int]): input height
        kernel_size ([int]): kernel size
        stride (int, optional): stride size. Defaults to 1.

    Returns:
        int: padding
    """
    size = (stride*input_size - stride - input_size + kernel_size)/2
    return math.floor(size)


class CnnSequence(nn.Module):  # param_count: 24 419 231; only classifier: 1631
    def __init__(self, num_labels, input_size, mode: str):
        super(CnnSequence, self).__init__()

        if(mode == 'hybrid'):
            self.in_channels = 2
        else:
            self.in_channels = 1
        # input-size (B, 2, 20,20)
        self.conv = nn.Conv2d(self.in_channels,
                              4,
                              3,
                              padding=same_padding(input_size, 3))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)

        # size (B, 4, 10,10)

        self.conv2 = nn.Conv2d(4,
                               8,
                               3,
                               padding=same_padding(input_size, 3))
        self.bn2 = nn.BatchNorm2d(8)
        # size (B, 8, 5,5)
        self.flat = nn.Flatten()
        #self.fc = nn.Linear(200, 100)  # 20*20*2
        self.drop = nn.Dropout(.2)
        self.fc1 = nn.Linear(200, num_labels)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))  # output_size B*4*20*20
        out = self.pool(out)  # output_size B*4*10*10
        out = self.relu(self.bn2(self.conv2(out)))  # output_size B*8*10*10
        out = self.pool(out)  # output_size B*8*5*5

        out = self.flat(out)  # output_size B*200
        out = self.drop(out)
        #out = self.fc(out)  # output_size B*100
        
        out = self.fc1(out)  # output_size B*num_lables
        return out


class CNN_For_Sequence(Seq2MatModelTemplate):
    def __init__(self, config):
        super(CNN_For_Sequence, self).__init__(config)
        if(config.num_labels < 1):
            config.num_labels = config.num_output_labels
        config.num_output_labels = config.num_labels
        self.num_labels = config.num_labels

        self.classifier = CnnSequence(
            self.num_labels,
            input_size=math.sqrt(config.embedding_size),
            mode=config.mode)


