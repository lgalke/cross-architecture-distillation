import json
from seq2mat import Seq2matConfig, Seq2matForSequenceClassification
from transformers import BertTokenizer

def test_cmow():
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Model
    with open('config/seq2mat_cmow.json') as fhandle:
        # ** unpacks dictionary into keyword arugments
        config = Seq2matConfig(**json.load(fhandle))
    classifier = Seq2matForSequenceClassification(config)

    # Tiny test
    tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
    outputs = classifier(input_ids=tokens)

def test_hybrid():
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Model
    with open('config/seq2mat_hybrid.json') as fhandle:
        # ** unpacks dictionary into keyword arugments
        config = Seq2matConfig(**json.load(fhandle))
    classifier = Seq2matForSequenceClassification(config)

    # Tiny test
    tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog")
    outputs = classifier(input_ids=tokens)


