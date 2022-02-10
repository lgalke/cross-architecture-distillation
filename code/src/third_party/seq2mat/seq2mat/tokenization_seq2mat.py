# coding=utf-8
# Copyright 2020 ANONYMIZED
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model Seq2mat."""


# import collections
import logging
import os
from shutil import copyfile

# from .tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

SPIECE_UNDERLINE = "▁"


logger = logging.getLogger(__name__)

####################################################
# In this template, replace all the XXX (various casings) with your model name
####################################################

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to file names for serializing Tokenizer instances
####################################################
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to pretrained vocabulary URL for all the model shortcut names.
####################################################
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "seq2mat-base": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
    }
}

####################################################
# Mapping from model shortcut names to max length of inputs
####################################################

# For some strange reason, this is necessary to re-use t5-spiece.model
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "seq2mat-base": 512,
}

####################################################
# Mapping from model shortcut names to a dictionary of additional
# keyword arguments for Tokenizer `__init__`.
# To be used for checkpoint specific configurations.
####################################################
PRETRAINED_INIT_CONFIGURATION = {
    "seq2mat-base": {},
}


# def load_vocab(vocab_file):
#     """Loads a vocabulary file into a dictionary."""
#     vocab = collections.OrderedDict()
#     with open(vocab_file, "r", encoding="utf-8") as reader:
#         tokens = reader.readlines()
#     for index, token in enumerate(tokens):
#         token = token.rstrip("\n")
#         vocab[token] = index
#     return vocab


class Seq2matTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a Seq2matTokenizer.
    :class:`~transformers.Seq2matTokenizer` runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_basic_tokenize=True
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        """Constructs a Seq2matTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
        """
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = Seq2matTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        # self.vocab = load_vocab(vocab_file)

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use T5Tokenizer:"
                "https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise

        self.do_basic_tokenize = False  # 2020-05-27
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        # return len(self.vocab)
        return self.sp_model.get_piece_size()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use T5Tokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text, sample=False):
        """ Take as input a string and return a list of strings (tokens) for words/sub-words
        """
        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        return pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
    #     """
    #     Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
    #     special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

    #     Args:
    #         token_ids_0: list of ids (must not contain special tokens)
    #         token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
    #             for sequence pairs
    #         already_has_special_tokens: (default False) Set to True if the token list is already formated with
    #             special tokens for the model

    #     Returns:
    #         A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
    #     """

    #     if already_has_special_tokens:
    #         if token_ids_1 is not None:
    #             raise ValueError(
    #                 "You should not supply a second sequence if the provided sequence of "
    #                 "ids is already formated with special tokens for the model."
    #             )
    #         return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

    #     if token_ids_1 is not None:
    #         return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    #     return [1] + ([0] * len(token_ids_0)) + [1]

    # def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
    #     """
    #     Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
    #     A BERT sequence pair mask has the following format:
    #     0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
    #     | first sequence    | second sequence

    #     if token_ids_1 is None, only returns the first portion of the mask (0's).
    #     """
    #     sep = [self.sep_token_id]
    #     cls = [self.cls_token_id]
    #     if token_ids_1 is None:
    #         return len(cls + token_ids_0 + sep) * [0]
    #     return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
