from transformers import (
    AlbertTokenizer,
    BertTokenizer,
    BertTokenizerFast
)


TOKENIZER = {
    "albert": lambda: (AlbertTokenizer.from_pretrained('albert-base-v2')),
    "bert": lambda: (BertTokenizer.from_pretrained("bert-base-uncased")),
    "bert-fast": lambda: (BertTokenizerFast.from_pretrained("bert-base-uncased")),
}


def load_tokenizer(identifier: str):
    if identifier in TOKENIZER:
        return TOKENIZER.get(identifier)()
    else:
        raise(ValueError("unknown tokenizer"))
