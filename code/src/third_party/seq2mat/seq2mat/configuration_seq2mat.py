# coding=utf-8
# Copyright 2020, ANONYMIZED
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
""" Seq2mat model configuration """


import logging

from transformers.configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

SEQ2MAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # "xxx-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/xxx-base-uncased-config.json",
    # "xxx-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/xxx-large-uncased-config.json",
}


class Seq2matConfig(PretrainedConfig):
    r"""
        :class:`~transformers.XxxConfig` is the configuration class to store the configuration of a
        `XxxModel`.


        Arguments:
            vocab_size: Vocabulary size of `inputs_ids` in `XxxModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `XxxModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = SEQ2MAT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "seq2mat"

    def __init__(
        self,
        mode='conv',
        vocab_size=30522,
        root_mat_size=20,
        embedding_size=400,
        initializer_mode='noisy-identity',
        initializer_range=0.01,
        embd_pdrop=0.1,
        hidn_pdrop=0.1,
        attn_pdrop=0.1,
        conv_stride=1,
        conv_kernel=3,
        conv_layers=5,
        pad_token_id=0,
        hidden_size=None,
        output_hidden_states=True,
        activation=None,
        skip_connections=False,
        layernorm=False,
        jumping_knowledge=False,
        intermediate_mlp=False,
        d_intermediate=None,
        attention=False,
        text_classifier=None,
        text_classifier_hidden_size=None,
        low_rank_approx=False,
        rec_iter=0,
        tied_weights=False,
        siamese=None,
        siamese_regression=None,
        bidirectional=False,
        # n_positions=1024,
        # n_ctx=1024,
        # n_embd=768,
        # n_layer=12,
        # n_head=12,
        # resid_pdrop=0.1,
        # embd_pdrop=0.1,
        # attn_pdrop=0.1,
        # layer_norm_epsilon=1e-5,
        # initializer_range=0.02,
        # summary_type="cls_index",
        # summary_use_proj=True,
        # summary_activation=None,
        # summary_proj_to_labels=True,
        # summary_first_dropout=0.1,
        num_labels=0,
        num_input_sequences=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.root_mat_size = root_mat_size
        self.embedding_size = embedding_size
        self.initializer_mode = initializer_mode
        self.initializer_range = initializer_range
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_layers = conv_layers
        # self.n_ctx = n_ctx
        # self.n_positions = n_positions
        # self.n_embd = n_embd
        # self.n_layer = n_layer
        # self.n_head = n_head
        # self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.hidn_pdrop = hidn_pdrop
        self.attn_pdrop = attn_pdrop


        self.output_hidden_states = output_hidden_states
        self.intermediate_mlp = intermediate_mlp
        self.attention = attention
        # self.attn_pdrop = attn_pdrop
        # self.layer_norm_epsilon = layer_norm_epsilon
        # self.summary_type = summary_type
        # self.summary_use_proj = summary_use_proj
        # self.summary_activation = summary_activation
        # self.summary_first_dropout = summary_first_dropout
        # self.summary_proj_to_labels = summary_proj_to_labels
        self.pad_token_id = pad_token_id
        if jumping_knowledge: assert output_hidden_states
        self.jumping_knowledge = jumping_knowledge



        # Siamese architectures
        if siamese: assert not jumping_knowledge
        if siamese: assert siamese in ["hadamard", "concat", "diffcat", "cosine", "mean"]
        if siamese_regression: assert siamese_regression in ["hadamard", "concat", "diffcat", "cosine", "mean"]

        self.siamese = siamese
        # Use same strategy for regression tasks as for other tasks if not specified
        self.siamese_regression = siamese_regression if siamese_regression else siamese



        assert mode in ['cmow', 'conv', 'hybrid', 'recursive'], "Unknown mode: "+mode
        if self.jumping_knowledge:
            assert mode == 'conv'
        if mode in ['cmow',  'hybrid']:
            assert self.conv_layers == 1, "Set conv_layers = 1 in cmow mode"
        if mode == 'recursive':
            assert self.embedding_size == self.root_mat_size
        self.mode = mode

        self.bidirectional = bool(bidirectional)
        if self.bidirectional:
            assert self.mode in ["cmow", "hybrid"]

        if hidden_size is not None:
            self._hidden_size = hidden_size
        else:
            if self.mode == 'hybrid':
                if self.bidirectional:
                    self._hidden_size = 2 * self.root_mat_size * self.root_mat_size + 2 * self.embedding_size
                else:
                    self._hidden_size = self.root_mat_size * self.root_mat_size + self.embedding_size
            elif self.mode == 'recursive':
                self._hidden_size = self.embedding_size
            elif self.mode == 'cmow':
                if self.bidirectional:
                    self._hidden_size = 2 * self.root_mat_size * self.root_mat_size
                else:
                    self._hidden_size = self.root_mat_size * self.root_mat_size
            elif self.mode == 'conv':
                self._hidden_size = self.root_mat_size * self.root_mat_size
            else:
                raise NotImplementedError("Unknown mode: "+self.mode)

        # Dimension for intermediate representations when using intermediate_mlp = True
        self.d_intermediate = d_intermediate

        self.skip_connections = skip_connections
        self.layernorm = layernorm
        # Coerce false / null to None
        self.activation = None if not activation else activation

        if tied_weights:
            # we can only tie if we have vector embeddings
            assert mode in ['hybrid' , 'recursive']
        self.tied_weights = tied_weights


        if text_classifier is None:
            text_classifier = 'linear'
        assert text_classifier in ['linear', 'mlp']
        self.text_classifier = text_classifier

        self.low_rank_approx = low_rank_approx


        if rec_iter:
            assert rec_iter > 0, "Rec iter must be falsy or positive integer"
            self.rec_iter = int(rec_iter)
        else:
            self.rec_iter = 0



        self.num_labels = num_labels
        self.num_input_sequences = num_input_sequences


    @property
    def max_position_embeddings(self):
        raise NotImplementedError
        return self.n_positions

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_attention_heads(self):
        raise NotImplementedError
        return self.n_head

    @property
    def num_hidden_layers(self):
        raise NotImplementedError
        return self.n_layer
