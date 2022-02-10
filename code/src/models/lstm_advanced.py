from src.third_party.seq2mat.seq2mat.modeling_seq2mat import Seq2matForSequenceClassification, Seq2matPreTrainedModel, Seq2matModel, flatten,   flatten_generic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss,  CrossEntropyLoss
from torch.autograd import Variable
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EasyLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, lstm_layers, num_classes, dropout, bidirectional=True):
        super(EasyLSTM, self).__init__()

        self.num_classes = num_classes

        self.dropout = nn.Dropout(.5)

        self.lstm = nn.LSTM(input_size=embedding_dim,  # The number of expected features in the input x
                            hidden_size=hidden_size,   # The number of features in the hidden state h
                            # number of lstm layers. 2 means there are two lstms in a row
                            num_layers=lstm_layers,
                            bidirectional=True,
                            batch_first=True)  # input -> (batch, seq, input_size)

        num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(hidden_size*num_directions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = 2
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size):
        h0 = torch.zeros(2, batch_size,
                         self.hidden_size).to(device)
        c0 = torch.zeros(2, batch_size,
                         self.hidden_size).to(device)
        return h0, c0

    def forward(self, x):
        batch_size = x.shape[0]
        #seq_len = [xx.size[0] for xx in x]
        h_0, c_0 = self.init_hidden(batch_size)

        # output_unpacked, output_lengths = pack_padded_sequence(
        # x,, batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self.lstm(x, (h_0, c_0))
        # packed_output shape = (batch, seq_len, num_directions * hidden_size)
        # hidden shape  = (num_layers * num_directions, batch, hidden_size)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rel = self.relu(cat)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds


class LSTM_Student(Seq2matForSequenceClassification):
    def __init__(self, config):
        super(LSTM_Student, self).__init__(config)
        if(config.num_labels < 1):
            config.num_labels = config.num_output_labels
        config.num_output_labels = config.num_labels
        self.num_labels = config.num_labels

        if config.mode == "hybrid":
            input_size = config.embedding_size*2
        else:
            input_size = config.embedding_size

        if config.siamese is not None and config.siamese:
            input_size *= 3

        self.classifier = EasyLSTM(
            input_size, input_size, 1, config.num_labels, .5)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        if self.siamese_model is not None:
            outputs = self.siamese_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        else:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        # --- outputs defined ---

        if self.mode == 'hybrid':
            clf_input = outputs[0]

        elif self.mode in ['cbow', 'cmow']:
            clf_input = outputs[0]
        elif self.mode == 'recursive':
            clf_input = outputs[1]  # Pooler output
        elif self.mode == 'conv':
            if self.jumping_knowledge:
                # Aggregate over layers for final prediction
                all_hidden_states = outputs[2]  # tuple of hidden states
                num_layers = len(all_hidden_states)
                h = torch.cat(all_hidden_states, dim=-1)
                # bsz x seqlen x d x d x layers
                h = h.mean(dim=1)
                clf_input = h.view(-1, self.d * self.d * num_layers)
            else:  # conv
                pooled_output = outputs[1]  # Pooler output
                clf_input = outputs[0]

        # if self.mode == 'hybrid':
        #     # Pooler outputs index 2 and 3 are relevant
        #     cmow_output = flatten_generic(outputs[2], -3 if self.bidirectional else -2)
        #     cbow_output = outputs[3]  # [bsz, (2,) d]
        #     if self.bidirectional:
        #         # fully-pooled CBOW outputs are same backward and forward -> only use one
        #         cbow_output = cbow_output[:, 0, :]
        #     clf_input = torch.cat([cmow_output, cbow_output], dim=-1)
        # elif self.mode == 'cmow':
        #     # Pooler output with index 1 is relevant
        #     clf_input = flatten_generic(outputs[1], -3 if self.bidirectional else -2)
        # elif self.mode == 'recursive':
        #     clf_input = outputs[1] # Pooler output
        # else:
        #     raise KeyError("Unknown mode")

        # CLF INPUT DEFINED

        # if self.layernorm:
        #     h = self.layernorm(h)
        if self.classifier is None:
            # Skip dropout when using only cosine similarity
            logits = pooled_output
        else:
            clf_input = self.dropout(clf_input)
            logits = self.classifier(clf_input)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

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
