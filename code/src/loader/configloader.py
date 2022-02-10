import json
import typing
from src.third_party.seq2mat.seq2mat import Seq2matConfig

folderpath = "src/third_party/seq2mat/config/"

# those are for: pretrained embeddings from the paper OR for training embeddings 'from scratch'
config_map = {
    "cmow": folderpath + "pretrained_paper/seq2mat_cmow.json",
    "conv": folderpath + "seq2mat_conv.json",
    "hybrid": folderpath + "pretrained_paper/seq2mat_hybrid.json",
    "recursive": folderpath + "seq2mat_recursive.json",
    "cbow": folderpath + "pretrained_paper/seq2mat_cbow.json"
}

# those are for: using ANONYMIZED's new pretrained embeddings with siamese config
ANONYMOUS_pretrained_config_map = {
    'hybrid': folderpath + '',  # todo
}


class Seq2MatConfigLenient(Seq2matConfig):
    @property
    def max_position_embeddings(self):
        return 0

    @property
    def num_attention_heads(self):
        return 0

    @property
    def num_hidden_layers(self):
        return 0


def load_config(config_id: str, num_labels: int, siamese: (str), bidirectional: (bool)) -> typing.Dict:
    path = config_map[config_id]
    with open(path) as fhandle:
        config = Seq2MatConfigLenient(**json.load(fhandle))
        config.num_labels = num_labels
        config.siamese = siamese
        config.num_input_sequences = 2 # checkme: is there any case in which we don't want this?!
        config.bidirectional = bidirectional
    if(config):
        return config
    else:
        raise(FileNotFoundError(f"{path} cannot be found"))
