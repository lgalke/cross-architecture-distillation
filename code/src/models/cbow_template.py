from torch import nn
from src.third_party.word2mat.word2mat import get_cbow_encoder


class CbowForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(CbowForSequenceClassification, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.encoder = get_cbow_encoder(self.config.vocab_size)
        self.classifier = nn.Sequential(
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):
        # get embeddings
        embeddings = self.encoder(input_ids)
        logits = self.classifier(embeddings)
        outputs = (logits,)
        # we get labels for the eval dataset! this needs to be taken care of
        # during training, fancyTrainer does the work for us
        # .. but here we have to return the loss for evaluation
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(outputs.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
