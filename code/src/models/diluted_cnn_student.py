import torch
import torch.nn as nn
import math
import pickle

from src.third_party.seq2mat.seq2mat import (
    Seq2matForSequenceClassification
)


def calculate_cnn_output_dim(h_in, w_in, stride, dilation, padding, kernel_size):
    a = (h_in+2 * padding[0] - dilation[0] * (kernel_size[0]-1)-1)
    h_out = math.floor((a/stride[0]) + 1)
    a = (w_in+2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1)
    w_out = math.floor((a / stride[1]) + 1)
    return (h_out, w_out)

def calculate_deconv_output_dim(h_in, w_in, stride, dilation, padding, kernel_size, output_padding):
    h_out = (h_in-1) * stride[0] - 2*padding[0] + dilation[0] * (kernel_size[0]-1)+output_padding[0]+1
    w_out = (w_in-1) * stride[1] - 2*padding[1] + dilation[1] * (kernel_size[1]-1)+output_padding[1]+1
    return (h_out, w_out)

class DilutedCnnStudent(Seq2matForSequenceClassification):
    def __init__(self, config):
        super(DilutedCnnStudent, self).__init__(config)
        # need a hack so we can restore our model using the transformer trainer etc.
        # config.num_labels is supposed to be an attribute with getter and setter, not a variable
        if(config.num_labels < 1):
            config.num_labels = config.num_output_labels
        config.num_output_labels = config.num_labels
        self.num_labels = config.num_labels

        if (config.mode == 'hybrid'):
            self.in_channels = 2
        else:
            self.in_channels = 1

        matrix_input_sz = int(math.sqrt(config.embedding_size))
        if config.siamese is not None and config.siamese:
            self.in_channels = 3
            if config.mode == 'hybrid':
                self.in_channels *= 2
                if config.bidirectional:
                    self.in_channels += 3


            #matrix_input_sz = int(math.sqrt(config.hidden_size*3))  # always assume diffcat

        # calculate dimensions before
        h1, w1 = calculate_deconv_output_dim(matrix_input_sz, matrix_input_sz, (4,4), (1,1), (0,0), (4,4), (0,0))
        h2, w2 = calculate_cnn_output_dim(h1, w1, (1,1), (1,1), (0,0), (4,4))
        h3, w3 = calculate_cnn_output_dim(h2, w2, (2,2), (1,1), (0,0), (3,3))
        h4, w4 = calculate_cnn_output_dim(h3, w3, (2,2), (1,1), (0,0), (3,3))

        self.classifier = nn.Sequential(
            # upscaling portion
            # scale by factor 4 -> 80x80
            torch.nn.Unflatten(1, (self.in_channels, matrix_input_sz, matrix_input_sz)),
            torch.nn.ConvTranspose2d(self.in_channels, self.in_channels, (4,4), stride=(4,4)),
            torch.nn.ReLU(),
            # one conv block
            torch.nn.Conv2d(self.in_channels, self.in_channels, (4,4), stride=1),
            torch.nn.Conv2d(self.in_channels, self.in_channels+2, (3,3), stride=2),
            torch.nn.Conv2d(self.in_channels+2, self.in_channels+2, (3,3), stride=2),
            torch.nn.BatchNorm2d(self.in_channels+2),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.4),
            nn.Linear(h4*w4*(self.in_channels+2), self.num_labels),
            nn.ReLU()
        )

    def load_pretrained_embeddings(self, filename):
        if self.mode != 'hybrid':
            with open(filename, 'rb') as f:
                weights = pickle.load(f)
            if weights is not None and weights['type'] == self.mode:
                for assignment in weights['assignments']:
                    # self.transformer.embeddings.matrix_embedding.embedding
                    self.transformer.embeddings.matrix_embedding.embedding.weight[
                        assignment['bert_index']].data.copy_(
                        torch.tensor(assignment['embedding']).view(-1).data)
        else:
            with open(filename, 'rb') as f:
                matrix_weights = pickle.load(f)
            with open(filename + '_cbow', 'rb') as f:
                cbow_weights = pickle.load(f)

            if matrix_weights is not None and matrix_weights['type'] == self.mode:
                for assignment in matrix_weights['assignments']:
                    # self.transformer.embeddings.matrix_embedding.embedding
                    self.transformer.embeddings.matrix_embedding.embedding.weight[assignment['bert_index']].data.copy_(
                        torch.tensor(assignment['embedding']).view(-1).data
                    )
            else:
                raise Exception("Hybrid Student: CMOW weights didn't load properly!")

            if cbow_weights is not None and cbow_weights['type'] == self.mode:
                for assignment in cbow_weights['assignments']:
                    # self.transformer.embeddings.matrix_embedding.embedding
                    self.transformer.embeddings.vector_embedding.weight[assignment['bert_index']].data.copy_(
                        torch.tensor(assignment['embedding']).view(-1).data
                    )
            else:
                raise Exception("Hybrid Student: CBOW weights didn't load properly!")

