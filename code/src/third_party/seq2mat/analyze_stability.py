import argparse
import json

import numpy as np
import torch
from transformers import BertTokenizer
from tqdm import tqdm

from seq2mat.modeling_seq2mat import Seq2matModel, Seq2matForSequenceClassification
from seq2mat.configuration_seq2mat import Seq2matConfig



parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to 1-sentence-per-line file")
parser.add_argument("--config", help="path to config file",
                    required=True)
parser.add_argument("--model_path", help="path to pretrained model",
                    default=None)
parser.add_argument("--batch_size", help="Batch size",
                    default=1, type=int)
parser.add_argument("--tokenizer_name", help="Tokenizer name",
                    default='bert-base-uncased')
args = parser.parse_args()

with open(args.config,'r') as config_file:
    config = Seq2matConfig(**json.load(config_file))

# Force to output all hidden states
config.output_hidden_states = True

def load_seq2mat_model(model_path, config,
                       cache_dir="/tmp/seq2mat/"):
    print("Loading Seq2mat model")
    model = Seq2matModel.from_pretrained(
        model_path,
        from_tf=False,
        config=config,
        cache_dir=cache_dir
    )
    return model


if args.model_path is None:
    print("Initializing blank model with config:", config)
    model = Seq2matModel(config)
else:
    print("Loading pretrained model from:", args.model_path, "using config:",
          config)
    model = load_seq2mat_model(args.model_path, config)

model.eval()
print(model)

d = config.root_mat_size
# print(model)


tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)


with open(args.path, 'r') as data_file:
    dataset = np.array([tokenizer.encode(line) for line in tqdm(data_file, desc="Loading data")])

def frobenius_list(input):
    """
    Takes [bsz, seqlen, d, d]
    Returns list of bsz*seqlen frobeniues norms
    """
    with torch.no_grad():
       norms = torch.norm(input, p='fro', dim=(-2,-1))
       norms = list(norms.view(-1).numpy())
    return norms

def det_list(input):
    with torch.no_grad():
        x = input.view(-1, input.size(-2), input.size(-1))
        dets = list(torch.det(x).numpy())
    return dets




pad_token_id = config.pad_token_id

def prepare_batch(seqs):
    maxlen = max(len(seq) for seq in seqs)
    padded_seqs = [seq + [config.pad_token_id] * (maxlen - len(seq))
                   for seq in seqs]
    return torch.LongTensor(padded_seqs)


pad_emb = model.embeddings.matrix_embedding.embedding.weight[pad_token_id].view(d,d)
with torch.no_grad():
    print("Frob norm of pad token: {:.4f}".format(torch.norm(pad_emb, p='fro', dim=(0,1)).item()))


num_layers = config.conv_layers
layer_frob_norms = {key: [] for key in range(num_layers+1)}
layer_dets = {key: [] for key in range(num_layers+1)}
layer_frob_norms['pooled'] = []
layer_dets['pooled'] = []


for i, start in tqdm(enumerate(range(0, len(dataset), args.batch_size)),
                     desc="Processing", total=len(dataset)):
    end = start + args.batch_size
    batch = dataset[start:end]
    input_ids = prepare_batch(batch)
    with torch.no_grad():
        embedding_output = model.embeddings(torch.LongTensor(input_ids))
        if config.mode == 'cmow':
            layer_frob_norms[0].extend(frobenius_list(embedding_output))
            layer_dets[0].extend(det_list(embedding_output))
            pooled_output = model.pooler(embedding_output)
            layer_frob_norms[1].extend(frobenius_list(pooled_output))
            layer_dets[1].extend(det_list(pooled_output))
        elif config.mode == 'conv':
            encoder_outputs = model(inputs_embeds=embedding_output)  # (final_hidden_state, pooler_output, all_hidden_states)
            all_hidden_states = encoder_outputs[2]
            for i in range(len(all_hidden_states)):
                layer_frob_norms[i].extend(frobenius_list(all_hidden_states[i]))
                layer_dets[i].extend(det_list(all_hidden_states[i]))

            # Pool the last hidden state
            pooled_output = encoder_outputs[1]
            layer_frob_norms['pooled'].extend(frobenius_list(pooled_output))
            layer_dets['pooled'].extend(det_list(pooled_output))


keys = list(range(num_layers+1))
keys.append('pooled')

for layer in keys:
    frobs = np.asarray(layer_frob_norms[layer])
    mean, std = frobs.mean(), frobs.std(ddof=1)
    print("#### Layer", layer, "####")
    print("\t- Frobenius Norm")
    print(f"\t\tMean = {mean:.4f} (SD = {std:.4f})")
    print("\t\tQuartiles: ({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})".format(*np.percentile(frobs, [0,25,50,75,100])))

    dets = np.asarray(layer_dets[layer])
    mean, std = dets.mean(), dets.std(ddof=1)
    print("\t- Determinants")
    print(f"\t\tMean = {mean:.4f} (SD = {std:.4f})")
    print("\t\tQuartiles: ({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f})".format(*np.percentile(dets, [0,25,50,75,100])))
    print("#" * 32)


