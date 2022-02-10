import torch
import torch.nn as nn
import pickle

from src.third_party.seq2mat.seq2mat import (
    Seq2matForSequenceClassification
)


class StudentForSequenceClassification(Seq2matForSequenceClassification):
    def __init__(self, config):
        super(StudentForSequenceClassification, self).__init__(config)
        # need a hack so we can restore our model using the transformer trainer etc.
        # config.num_labels is supposed to be an attribute with getter and setter, not a variable
        if(config.num_labels < 1):
            config.num_labels = config.num_output_labels
        config.num_output_labels = config.num_labels
        self.num_labels = config.num_labels

        half_size = int(config.hidden_size/2)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(config.hidden_size),
            nn.Linear(config.hidden_size, half_size),
            nn.ReLU(half_size),
            torch.nn.LayerNorm(half_size),
            nn.Linear(half_size, self.num_labels)
        )

    def load_pretrained_embeddings(self, filename):
        if self.mode != 'hybrid':
            with open(filename, 'rb') as f:
                weights = pickle.load(f)
            if weights is not None and weights['type'] == self.mode:
                for assignment in weights['assignments']:
                    # self.transformer.embeddings.matrix_embedding.embedding
                    if self.mode == 'cbow':
                        self.transformer.embeddings.vector_embedding.weight[assignment['bert_index']].data.copy_(
                            torch.tensor(assignment['embedding']).view(-1).data)
                    else:
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
                raise Exception(
                    "Hybrid Student: CMOW weights didn't load properly!")

            if cbow_weights is not None and cbow_weights['type'] == self.mode:
                for assignment in cbow_weights['assignments']:
                    # self.transformer.embeddings.matrix_embedding.embedding
                    self.transformer.embeddings.vector_embedding.weight[assignment['bert_index']].data.copy_(
                        torch.tensor(assignment['embedding']).view(-1).data
                    )
            else:
                raise Exception(
                    "Hybrid Student: CBOW weights didn't load properly!")
