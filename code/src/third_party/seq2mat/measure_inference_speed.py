from timeit import default_timer as timer
import json

import numpy as np
import torch

from transformers import AutoConfig, AutoModel, AutoTokenizer
from allennlp.modules import Elmo

from seq2mat import Seq2matModel, Seq2matConfig

NUM_BATCHES = 1024 # How often to repeat the encoding
BATCH_SIZE = 256 # Number of sequences in a batch
SEQLEN = 64  # Length of sequences
VOCAB_SIZE = 10000  # Doesn't matter, should be less than true model vocab size


NUM_INPUT_SEQUENCES = 1
NUM_LABELS = 2

CACHE_DIR = '/tmp/seq2mat-infspeed'

DUMMY_TASK_NAME = "cola"  # Some single-sentence task


SEQ2MAT_CONFIG = "config/seq2mat_hybrid_bidirectional_diffcat.json"
SEQ2MAT_PATH = "zoo/seq2mat_hybrid_bidirectional_sbertlike-100p-bsz512/"

DEVICE = torch.device("cuda:0")


def build_model(model_name):
    model = None
    if model_name == 'seq2mat':
        with open(SEQ2MAT_CONFIG, 'r') as fh:
            config = Seq2matConfig(**json.load(fh),
                                   num_labels=NUM_LABELS,
                                   num_input_sequences=NUM_INPUT_SEQUENCES)
        # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME,
        #                                           cache_dir=CACHE_DIR)
        model = Seq2matModel.from_pretrained(
            SEQ2MAT_PATH,
            from_tf=False,
            config=config,
            cache_dir=CACHE_DIR
        )
    elif model_name == 'elmo':
        model = Elmo("config/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                     "zoo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                     2, # num_output_representations
                     requires_grad=False
                     )
    else:
        # Else case: some model from the huggingface vocabulary
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            finetuning_task=DUMMY_TASK_NAME,
            cache_dir=CACHE_DIR,
        )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_name,
        #     cache_dir=CACHE_DIR,
        # )
        model = AutoModel.from_pretrained(
            model_name,
            from_tf=False,
            config=config,
            cache_dir=CACHE_DIR,
        )
    return model


def generate_random_sequences(vocab_size, num_batches, batch_size, seqlen,
                              for_elmo=False):
    if for_elmo:
        sequences = torch.randint(20, (num_batches, batch_size, seqlen, 50))
    else:
        sequences = torch.randint(vocab_size,
                                  (num_batches, batch_size, seqlen))
    return sequences


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model type: seq2mat | elmo | bert-base-uncased | distilbert-base-uncased | google/mobilebert-uncased ...")
    args = parser.parse_args()

    print("Loading model...")
    model = build_model(args.model_name).to(DEVICE)

    # Generate random sequences
    DATA = generate_random_sequences(VOCAB_SIZE,
                                     NUM_BATCHES,
                                     BATCH_SIZE,
                                     SEQLEN,
                                     for_elmo=(args.model_name == 'elmo')).to(DEVICE)

    model.requires_grad = False # *inference*

    print("Measuring inference time...")
    with torch.no_grad(): # *inference*
        if args.model_name == 'elmo':
            start = timer()
            # CRITICAL CODE begin
            for i in range(NUM_BATCHES):
                __outputs = model(inputs=DATA[i])
            # CRITICAL CODE end
            end = timer()
        else:
            start = timer()
            # CRITICAL CODE begin
            for i in range(NUM_BATCHES):
                __outputs = model(input_ids=DATA[i])
            # CRITICAL CODE end
            end = timer()

    N = NUM_BATCHES * BATCH_SIZE

    seconds = end - start

    sentences_per_second = N / seconds

    print("Took", end-start, "seconds.")
    print("Encoded", sentences_per_second, "sentences per second.")
    print("Num params:", count_params(model))

if __name__ == '__main__':
    main()
