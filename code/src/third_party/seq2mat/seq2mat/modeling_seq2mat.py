# coding=utf-8
# Copyright 2018 ANONYMIZED
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
""" PyTorch seq2mat model. """

####################################################
# In this template, replace all the XXX (various casings) with your model name
####################################################


import logging
import os

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from .configuration_seq2mat import Seq2matConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import PreTrainedModel

from .matmul_pooling import matmul_pool, pad_identity
from .recursive import ThreeWayMVComposition


logger = logging.getLogger(__name__)

####################################################
# This dict contrains shortcut names and associated url
# for the pretrained weights provided with the models
####################################################
# SEQ2MAT_PRETRAINED_MODEL_ARCHIVE_MAP = {
#     "seq2mat-base-uncased": "https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin",
#     "seq2mat-large-uncased": "https://cdn.huggingface.co/bert-large-uncased-pytorch_model.bin",
# }


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_seq2mat(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (itself a sub-class of torch.nn.Module)
####################################################

####################################################
# Here is an example of typical layer in a PyTorch model of the library
# The classes are usually identical to the TF 2.0 ones without the 'TF' prefix.
#
# See the conversion methods in modeling_tf_pytorch_utils.py for more details
####################################################


class Seq2matAttention(nn.Module):
    def __init__(self, config):
        super(Seq2matAttention, self).__init__()
        # Deprecated code, hope it didn't corrupt checkpoints
        # If not: TODO delete it
        # self.attention = nn.Sequential(
        #     nn.Linear(3 * config.root_mat_size,
        #               config.hidden_size),
        #     nn.ReLU(config.hidden_size),
        #     nn.Dropout(config.attn_pdrop),
        #     nn.Linear(config.hidden_size, 3)
        # )
        self.stride = config.conv_stride
        self.kernel = config.conv_kernel
        self.d = config.root_mat_size
        self.dense = Linear2D(self.d)
        self.attn = nn.Sequential(
            nn.Linear(self.d * self.d * self.kernel, self.kernel),
            nn.LeakyReLU(),
            nn.Softmax(dim=2),
            nn.Dropout(config.attn_pdrop)
        )

    def forward(self, inputs):
        # Pad = kernel_size  -1 such that len(output) == len(input)
        d, pad = self.d, self.kernel - 1
        # Left -> round down || Right -> round up
        # Odd kernel size leads to equal padding on both sides
        pad_left, pad_right = pad // 2, (pad - 1) // 2 + 1

        bsz = inputs.size(0)

        h = self.dense(inputs)
        h_pad = pad_identity(h, pad_left, pad_right)
        h_unf = h_pad.unfold(1, self.kernel, self.stride)
        # x_unf : [bsz, n_chunks, d, d, kernel_size]
        attention_inputs = h_unf.reshape(bsz, h_unf.size(1),
                                         d * d * h_unf.size(-1))
        # attention_inputs: [ bsz, n_chunks, d*d*kernel_size ]

        # match shape of h_unf
        alpha = self.attn(attention_inputs).view(bsz, h_unf.size(1), 1, 1,
                                                 h_unf.size(-1))
        # alpha: scalars [ bsz, n_chunks, 1, 1, kernel_size]

        # Apply attention coefficients
        h_attn = alpha * h_unf

        # Matmul all the things
        outputs =  h_attn[:, :, :, :, 0]
        for i in range(1, h_attn.size(4)):  # Go over kernel size
            # Matmul within last two dims
            outputs = torch.einsum('bcij,bcjk->bcik', outputs,
                                   h_attn[:, :, :, :, i])
        return outputs


class Linear2D(nn.Linear):
    """ Linear on flattened 2D Tensors, returning 2D tensor of same size"""
    def __init__(self, root_mat_size, bias=True):
        super(Linear2D, self).__init__(root_mat_size, root_mat_size, bias=bias)

    def forward(self, input):
        size = input.size()
        x = flatten(input)
        h = super().forward(x)
        return h.view(*size[:-2], size[-2], size[-1])


class Seq2matDenseReluDense(nn.Module):
    """ Seq2mat intermediate layer transformation"""
    def __init__(self, config):
        super().__init__()
        self.d = config.root_mat_size

        if config.d_intermediate:
            # Use specified intermediate size, if present
            d_hidden = config.d_intermediate
        else:
            # Else use d^2 (same-size as input)
            d_hidden = self.d * self.d

        self.wi = nn.Linear(self.d * self.d, d_hidden)
        self.wo = nn.Linear(d_hidden, self.d * self.d)
        self.dropout = nn.Dropout(config.hidn_pdrop)


    def forward(self, inputs):
        x = flatten(inputs)
        h = self.wi(x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        # Restore matrix representations in last two dimensions
        return h.view(*h.size()[:-1], self.d, self.d)


# class Seq2mat(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention = Seq2matAttention(config)
#         self.intermediate = Seq2matIntermediate(config)
#         self.output = Seq2matOutput(config)

#     def forward(self, hidden_states, attention_mask=None, head_mask=None):
#         attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
#         attention_output = attention_outputs[0]
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
#         return outputs


####################################################
# PreTrainedModel is a sub-class of torch.nn.Module
# which take care of loading and saving pretrained weights
# and various common utilities.
#
# Here you just need to specify a few (self-explanatory)
# pointers for your model and the weights initialization
# method if its not fully covered by PreTrainedModel's default method
####################################################

Seq2matLayerNorm = torch.nn.LayerNorm


# class Seq2matMatrixEmbedding_DEPRECATED(nn.Module):
#     """ Embeds each token as a weight *matrix* """
#     def __init__(self, config):
#         super(Seq2matMatrixEmbedding, self).__init__()
#         self.padding_idx = config.pad_token_id
#         self.initializer_range = config.initializer_range
#         d = self.root_mat_size = config.root_mat_size
#         # Todo add matrix factorization option
#         self.weight = nn.Parameter(torch.FloatTensor(config.vocab_size, d, d))

#     def reset_parameters(self):
#         d = self.root_mat_size
#         nn.init.normal_(self.weight.data, mean=0.0, std=self.initializer_range)
#         self.weight.data += torch.eye(d, d)  # broadcasts
#         if self.padding_idx is not None:
#             with torch.no_grad():
#                 self.weight.data[self.padding_idx] = torch.eye(d, d)

#     def forward(self, input_ids):
#         """ For input LongTensor of shape [bsz, seqlen],
#         Return:
#             Embedding FloatTensor of shape [bsz, seqlen, d, d]
#         """
#         return torch.embedding(self.weight, input_ids)


class Seq2matMatrixEmbedding(nn.Module):
    def __init__(self, config):
        super(Seq2matMatrixEmbedding, self).__init__()
        self.padding_idx = config.pad_token_id
        self.initializer_mode = config.initializer_mode
        self.initializer_range = config.initializer_range
        d = self.root_mat_size = config.root_mat_size

        self.factorized = bool(config.low_rank_approx)

        if self.factorized:
            self.r = int(config.low_rank_approx)
            self.emb_l = nn.Embedding(config.vocab_size, d*self.r)
            self.emb_r = nn.Embedding(config.vocab_size, d*self.r)
            self.emb_diag = nn.Embedding(config.vocab_size, d)
        else:
            self.embedding = nn.Embedding(config.vocab_size, d*d)


    def reset_parameters(self):
        d = self.root_mat_size
        if self.factorized:
            if self.initializer_mode == 'noisy-identity':
                nn.init.normal_(self.emb_l.weight.data,
                                mean=0.0, std=self.initializer_range)
                nn.init.normal_(self.emb_r.weight.data,
                                mean=0.0, std=self.initializer_range)
                nn.init.normal_(self.emb_diag.weight.data,
                                mean=1.0, std=self.initializer_range)

                if self.padding_idx is not None:
                    with torch.no_grad():
                        self.emb_l.weight.data[self.padding_idx] = 0.
                        self.emb_r.weight.data[self.padding_idx] = 0.
                        self.emb_diag.weight.data[self.padding_idx] = 1.


            else:
                raise NotImplementedError("""
                                          Initializer mode not available in
                                          low-rank approximation case
                                          """)


        else:
            if self.initializer_mode == 'noisy-identity':
                nn.init.normal_(self.embedding.weight.data,
                                mean=0.0, std=self.initializer_range)
                with torch.no_grad():
                    self.embedding.weight.data += torch.eye(d, d).view(1, d*d) # broadcasts
                if self.padding_idx is not None:
                    with torch.no_grad():
                        self.embedding.weight.data[self.padding_idx] = torch.eye(d, d).view(1, d*d).contiguous()
            elif self.initializer_mode == 'identity':
                self.embedding.weight.data = torch.eye(d,d)\
                    .expand(self.embedding.weight.data.size(0), d, d).view(-1, d*d).contiguous()
            elif self.initializer_mode == 'normal':
                nn.init.normal_(self.embedding.weight.data,
                                mean=0.0, std=self.initializer_range)
                if self.padding_idx is not None:
                    with torch.no_grad():
                        self.embedding.weight.data[self.padding_idx] = torch.eye(d, d).view(1, d*d).contiguous()
        # print(self.embedding.weight[0].view(d, d))
        # print(self.embedding.weight[42].view(d, d))
        # input()

    def forward(self, input_ids):
        d = self.root_mat_size
        if self.factorized:
            r = self.r
            emb_l = self.emb_l(input_ids)
            emb_r = self.emb_r(input_ids)

            emb_l = emb_l.view(*emb_l.size()[:-1], d, r)
            emb_r = emb_r.view(*emb_r.size()[:-1], r, d)
            emb_diag = self.emb_diag(input_ids)

            emb = emb_l @ emb_r # '@' is more flexible than bmm
            emb = emb + torch.diag_embed(emb_diag)
        else:
            emb = self.embedding(input_ids)
            emb = emb.view(*emb.size()[:-1], d, d)
        return emb.contiguous()



class Seq2matEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bidirectional = config.bidirectional
        if config.layernorm and (config.mode =='hybrid' or config.bidirectional):
            raise NotImplementedError("Layernorm with hybrid model not implemented")
        self.matrix_embedding = Seq2matMatrixEmbedding(config)
        if self.bidirectional:
            self.backward_matrix_embedding = Seq2matMatrixEmbedding(config)
        if config.mode in ['hybrid', 'recursive']:
            self.vector_embedding = nn.Embedding(config.vocab_size,
                                                 config.embedding_size,
                                                 padding_idx=config.pad_token_id)
            self.is_hybrid = True
        else:
            self.is_hybrid = False
        self.dropout = nn.Dropout(config.embd_pdrop)

        if config.layernorm:
            self.layernorm = Seq2matLayerNorm((config.root_mat_size,
                                               config.root_mat_size),
                                              elementwise_affine=True)
            if self.bidirectional:
                self.backward_layernorm = Seq2matLayerNorm((config.root_mat_size,
                                                            config.root_mat_size),
                                                           elementwise_affine=True)
        else:
            self.layernorm = None

    def forward(self, input_ids, **kwargs):
        """ Returns (mat_embeddings, [vec_embeddings]) for input_ids """
        mat_embeddings = self.matrix_embedding(input_ids)
        mat_embeddings = self.dropout(mat_embeddings)
        if self.layernorm:
            mat_embeddings = self.layernorm(mat_embeddings)
        outputs = (mat_embeddings,)

        if self.bidirectional:
            bw_mat_embeddings = self.backward_matrix_embedding(input_ids)
            bw_mat_embeddings = self.dropout(bw_mat_embeddings)
            if self.layernorm:
                bw_mat_embeddings = self.backward_layernorm(bw_mat_embeddings)
            outputs += (bw_mat_embeddings,)

        if self.is_hybrid:
            vec_embeddings = self.vector_embedding(input_ids)
            vec_embeddings = self.dropout(vec_embeddings)
            outputs = outputs + (vec_embeddings,)

        #                               if bidirectional     if hybrid or recursive
        # outputs : (forward_mat_emb, [ backward_mat_emb, ] [vec_emb]
        return outputs


class Seq2matLayer(nn.Module):
    def __init__(self, config):
        super(Seq2matLayer, self).__init__()
        self.kernel = config.conv_kernel
        self.stride = config.conv_stride

        self.skip_connections = config.skip_connections

        self.dropout = nn.Dropout(config.hidn_pdrop)


        if not config.activation:
            self.activation = None
        elif callable(config.activation):
            self.activation = config.activaiton
        elif isinstance(config.activation, str):
            self.activation = getattr(torch.nn.functional,
                                    config.activation)
        else:
            raise ValueError("Unkown activation:", config.activation)

        if config.intermediate_mlp:
            self.mlp = Seq2matDenseReluDense(config)
        else:
            self.mlp = None

        if config.layernorm:
            self.layernorm1 = Seq2matLayerNorm((config.root_mat_size, config.root_mat_size), elementwise_affine=True)
            if self.mlp:
                self.layernorm2 = Seq2matLayerNorm((config.root_mat_size, config.root_mat_size), elementwise_affine=True)
        else:
            self.layernorm1 = None
            self.layernorm2 = None

        if config.attention:
            self.attention = Seq2matAttention(config)
        else:
            self.attention = None

    def forward(self, x):
        ################
        # MATMUL BLOCK #
        ################
        matmul_in = self.layernorm1(x) if self.layernorm1 else x

        if self.attention:
            # Attention module knows kernel and stride from config
            h = self.attention(matmul_in)

        else:
            h = matmul_pool(matmul_in, self.kernel, self.stride)

        if self.activation is not None:
            h = self.activation(h)

        h = self.dropout(h)

        # if self.skip_connections:
        #     h = x + h

        #############
        # MLP BLOCK #
        #############

        if self.mlp:
            mlp_in = self.layernorm2(h) if self.layernorm2 else h
            mlp_out = self.mlp(mlp_in)
            if self.skip_connections:
                h = h + mlp_out
            else:
                h = mlp_out


        return h


class Seq2matRecursiveEncoder(nn.Module):
    def __init__(self, config):
        super(Seq2matRecursiveEncoder, self).__init__()
        self.output_hidden_states = config.output_hidden_states
        self.composition = ThreeWayMVComposition(config.root_mat_size,
                                                 config.activation,
                                                 config.activation)
        self.residual = bool(config.skip_connections)
        self.rec_iter = config.rec_iter

    def forward(self, x_vec, x_mat):
        all_hidden_states = ()
        h_vec, h_mat = x_vec, x_mat

        if self.rec_iter:
            n = self.rec_iter
        else:
            n = h_vec.size(1) # if recursive iterations not specified,
            # use as many as sequence length

        for __ in range(n):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + ((h_vec, h_mat),)
            z_vec, z_mat = self.composition(h_vec, h_mat)

            if self.residual:
                h_vec = h_vec + z_vec
                h_mat = h_mat + z_mat
            else:
                h_vec, h_mat = z_vec, z_mat

        # Final state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + ((h_vec, h_mat),)

        outputs = ((h_vec, h_mat),)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        return outputs


class Seq2matEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_hidden_states = config.output_hidden_states
        self.layers = nn.ModuleList([Seq2matLayer(config) for _
                                     in range(config.conv_layers)])

    def forward(self, hidden_states):
        """ hiddens x of shape [batch_size, d, d] """
        all_hidden_states = ()
        for i, layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

def flatten(x):
    """ Flattens [b, l, d, d] to [b, l, d * d]"""
    size = x.size()
    # Safe flatten for multiple dimensions!
    return x.view(*size[:-2], size[-2] * size[-1])

def flatten_generic(x, dim=-2):
    """ Flattens a generic number of trailing dimensions,
    dim=-2 -> flattens last two dimensions
    dim=-3 -> flattens last three dimensions
    -- Will replace function above eventually --
    """
    size = x.size()
    return x.view(*size[:dim], np.product(size[dim:]))


class Seq2matCMOW(nn.Module):
    """
    Continual Multiplication of Words --
    Composes multiple embedding matrices into a single matrix:
    input_embeds: [bsz,seqlen,d,d]
    output: [bsz,d,d], [bsz, seqlen, d, d]
    """
    def __init__(self, reverse=False, flatten=False):
        super().__init__()
        self.reverse = reverse
        self.flatten = flatten

    def forward(self, input_embeds=None, **kwargs):
        """Input_embeds of shape [batch_size, seq_len, d, d]"""
        bsz, seqlen, d, __ = input_embeds.size()
        if self.reverse:
            # go backward direction
            start = seqlen - 1
            index_iterator = range(seqlen - 2, -1, -1)
        else:
            start = 0
            index_iterator = range(1, seqlen)

        x = input_embeds[:, start, :, :]

        all_outputs = [x]

        for i in index_iterator:
            x = torch.bmm(x, input_embeds[:, i, :, :])
            all_outputs.append(x)

        all_outputs = torch.stack(all_outputs, dim=1)

        if self.flatten:
            x = flatten_generic(x, dim=-2)
            all_outputs = flatten_generic(all_outputs, dim=-2)

        return x, all_outputs



class Seq2matCBOW(nn.Module):
    """
    Continual Multiplication of Words --
    Composes multiple embedding matrices into a single matrix:
    input_embeds: [bsz,seqlen,d,d]
    output: [bsz,d,d], [bsz, seqlen, d, d]
    """
    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    def forward(self, input_embeds=None, **kwargs):
        """Input_embeds of shape [batch_size, seq_len, d_vec]"""
        bsz, seqlen, d = input_embeds.size()

        if self.reverse:
            # go backward direction
            start = seqlen - 1
            index_iterator = range(seqlen - 2, -1, -1)
        else:
            start = 0
            index_iterator = range(1, seqlen)

        x = input_embeds[:, start, :]

        all_outputs = [x]

        for i in index_iterator:
            x = x + input_embeds[:, i, :]
            all_outputs.append(x)

        return x, torch.stack(all_outputs, dim=1)



class Seq2matPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = Seq2matConfig
    # pretrained_model_archive_map = SEQ2MAT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_seq2mat
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        # print("="*40+"> _init_weights called")
        """ Initialize the weights """
        if isinstance(module, Seq2matMatrixEmbedding):
            # Inits according to initializer range, init mode
            print("init matrix embedding")
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            print("init linear or embedding")
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Seq2matLayerNorm):
            print("init layernorm")
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            print("init bias of linears")
            module.bias.data.zero_()


SEQ2MAT_START_DOCSTRING = r"""    The SEQ2MAT model was proposed in
    `SEQ2MAT: <++TITLE++>`_
    by <++AUTHORS++}. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`XXX: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.Seq2matConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

SEQ2MAT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, Seq2mat input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Seq2mat is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.Seq2matTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `Seq2mat: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare seq2mat Model transformer outputting raw hidden-states without any specific head on top.",
    SEQ2MAT_START_DOCSTRING,
    SEQ2MAT_INPUTS_DOCSTRING,
)
class Seq2matModel(Seq2matPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during seq2mat pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = Seq2matTokenizer.from_pretrained('seq2mat-base-uncased')
        model = Seq2matModel.from_pretrained('seq2mat-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super().__init__(config)
        self.mode = config.mode
        self.embeddings = Seq2matEmbeddings(config)

        if self.mode == 'recursive':
            self.encoder = Seq2matRecursiveEncoder(config)
            self.cmow, self.cbow = None, None
        elif self.mode == 'hybrid':
            self.encoder = None
            self.cbow = Seq2matCBOW(reverse=False)
            self.cmow = Seq2matCMOW(reverse=False, flatten=True)
            if config.bidirectional:
                self.cbow_bw = Seq2matCBOW(reverse=True)
                self.cmow_bw = Seq2matCMOW(reverse=True, flatten=True)
        elif self.mode == 'cmow':
            self.encoder = None
            self.cbow = None
            self.cmow = Seq2matCMOW(reverse=False, flatten=True)
            if config.bidirectional:
                self.cbow_bw = None
                self.cmow_bw = Seq2matCMOW(reverse=True, flatten=True)
        elif self.mode == 'conv':
            self.encoder = Seq2matEncoder(config)
            self.cmow, self.cbow = None, None
        else:
            raise NotImplementedError("Unknown mode: "+ self.mode)


        self.config = config

        self.init_weights()

    def get_input_embeddings(self):
        print("Get input embeddings called")
        # raise NotImplementedError("Not properly implemented for all cases")
        return self.embeddings.vector_embedding

    def set_input_embeddings(self, new_embeddings):
        print("Set input embeddings called")
        # raise NotImplementedError("Not properly implemented for all cases")
        self.embeddings.vector_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError("Seq2mat does not support pruning")
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # elif input_ids is not None:
        #     input_shape = input_ids.size()
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        # else:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")

        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        # device = input_ids.device

        # if attention_mask is None:
        #     attention_mask = torch.ones(input_shape, device=device)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # # We create a 3D attention mask from a 2D tensor mask.
        # # (this can be done with self.invert_attention_mask)
        # # Sizes are [batch_size, 1, 1, to_seq_length]
        # # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # # this attention mask is more simple than the triangular masking of causal attention
        # # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # # masked positions, this operation will create a tensor which is 0.0 for
        # # positions we want to attend and -10000.0 for masked positions.
        # # Since we are adding it to the raw scores before the softmax, this is
        # # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # # Prepare head mask if needed
        # # 1.0 in head_mask indicate we keep the head
        # # attention_probs has shape bsz x n_heads x N x N
        # # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        ##################################
        # Replace this with your model code
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids=input_ids)

        # Disassemble different embeddings

        if self.mode == 'conv':
            mat_embeds = inputs_embeds[0]
            encoder_outputs = self.encoder(mat_embeds)
            sequence_output = encoder_outputs[0]  # [bsz, len, d, d]

            sequence_output = flatten_generic(sequence_output, dim=-2)  # [bsz, len, d*d]

            pooled_output = sequence_output.mean(dim=1)

            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
            # output is flat
        elif self.mode == 'cmow':
            mat_embeds = inputs_embeds[0]
            pooled_output, all_outputs = self.cmow(mat_embeds)
            # pooled_output: [bsz, d * d] (already flattened)
            # all_outputs: [bsz, len, d * d] (already flattened)
            if self.config.bidirectional:
                mat_embeds_bw = inputs_embeds[1]
                pooled_output_bw, all_outputs_bw = self.cmow_bw(mat_embeds_bw)
                outputs = (
                    # torch.stack([all_outputs, all_outputs_bw], dim=2),
                    # torch.stack([pooled_output, pooled_output_bw], dim=1),
                    torch.cat([all_outputs, all_outputs_bw], dim=-1),
                    torch.cat([pooled_output, pooled_output_bw], dim=-1),
                    (mat_embeds, mat_embeds_bw)  # <- tuple
                )
            else:
                # No bidirection
                outputs = (all_outputs, pooled_output, (mat_embeds,))
        elif self.mode == 'hybrid':
            mat_embeds = inputs_embeds[0]
            vec_embeds = inputs_embeds[-1] # -1 because there could be backward embeddings
            pooled_cmow_output, all_cmow_outputs = self.cmow(mat_embeds)
            pooled_cbow_output, all_cbow_outputs = self.cbow(vec_embeds)

            # Concat CMOW and CBOW outputs
            pooled_output = torch.cat([pooled_cbow_output, pooled_cmow_output], dim=-1)
            all_outputs = torch.cat([all_cbow_outputs, all_cmow_outputs], dim=-1)

            if self.config.bidirectional:
                mat_embeds_bw = inputs_embeds[1] # backward mat embeddings
                pooled_cmow_output_bw, all_cmow_outputs_bw = self.cmow_bw(mat_embeds_bw)
                # Use same vector embeddings as in forward pass
                pooled_cbow_output_bw, all_cbow_outputs_bw = self.cbow_bw(vec_embeds)

                # Concat CMOW and CBOW outputs
                # pooled_output_bw = torch.cat([pooled_cbow_output_bw, pooled_cmow_output_bw], dim=-1)

                # Concat Forward and backward outputs
                # (for pooled CBOW, fw is same as bw, therefore use only 1
                pooled_output = torch.cat([pooled_output, pooled_cmow_output_bw], dim=-1)

                all_outputs_bw = torch.cat([all_cbow_outputs_bw, all_cmow_outputs_bw], dim=-1)
                all_outputs = torch.cat([all_outputs, all_outputs_bw], dim=-1)

                outputs = (all_outputs, pooled_output, (mat_embeds, mat_embeds_bw, vec_embeds))
#                 outputs = (
#                     torch.stack([all_cmow_outputs, all_cmow_outputs_bw],
#                                 dim=2),  # 0
#                     torch.stack([all_cbow_outputs, all_cbow_outputs_bw],
#                                 dim=2),  # 1
#                     torch.stack([pooled_cmow_output, pooled_cmow_output_bw],
#                                 dim=1),  # 2
#                     torch.stack([pooled_cbow_output, pooled_cbow_output_bw],
#                                 dim=1),  # 3
#                     (mat_embeds, mat_embeds_bw, vec_embeds))  # 4
            else:
                outputs = (all_outputs, pooled_output, (mat_embeds, vec_embeds))

        elif self.mode == 'recursive':
            mat_embeds, vec_embeds = inputs_embeds
            encoder_outputs = self.encoder(vec_embeds, mat_embeds)
            sequence_output = encoder_outputs[0] # (vec, mat)-tuple
            sequence_output = sequence_output[0] # only use vectors
            pooled_output = sequence_output.mean(dim=1)
            outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
            # output is flat

        else:
            raise KeyError("Unknown mode: " + self.mode)

        # Output: #

        return outputs  # sequence_output, (pooled_output), (hidden_states), (attentions)


def siamese_hadamard(a, b):
    return a * b

def siamese_concat(a, b):
    return torch.cat([a,b], dim=-1)

def siamese_concat_diff(a, b):
    return torch.cat([a, b, (a-b).abs()], dim=-1)

def siamese_cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1)

SIAMESE_FUNCTION_MAP = {
    'hadamard': siamese_hadamard,
    'concat': siamese_concat,
    'diffcat': siamese_concat_diff,
    'cosine': siamese_cosine_similarity,
}

class Seq2matSiamese():
    """ Wrapper to apply the same model to each input separately and then combine outputs"""
    def __init__(self, model, combine_fn):
        self.model = model
        self.combine_fn = combine_fn if callable(combine_fn) else SIAMESE_FUNCTION_MAP[combine_fn]

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        # Unpack based on token type ids
        a_kwargs, b_kwargs = self._unpack(token_type_ids, input_ids=input_ids, inputs_embeds=inputs_embeds)
        # Apply same model on two sequences
        a_outputs = self.model(**a_kwargs)
        b_outputs = self.model(**b_kwargs)
        # Pack together by using combine_fn
        outputs = self._pack(a_outputs, b_outputs)
        return outputs

    def _unpack(self, token_type_ids,
               input_ids=None,
               attention_mask=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None):
        """ Unpacks input arguments into two dicts, depending on token_type_ids """
        if input_ids is not None:
            a_input_ids = input_ids * (1 - token_type_ids)
            b_input_ids = input_ids * token_type_ids
        else:
            a_input_ids = None
            b_input_ids = None

        if attention_mask is not None:
            a_attention_mask = attention_mask * (1 - token_type_ids)
            b_attention_mask = attention_mask * token_type_ids
        else:
            a_attention_mask = None
            b_attention_mask = None

        if position_ids is not None:
            a_position_ids = position_ids * (1 - token_type_ids)
            b_position_ids = position_ids * token_type_ids
        else:
            a_position_ids = None
            b_position_ids = None

        if head_mask is not None:
            a_head_mask = head_mask * (1 - token_type_ids)
            b_head_mask = head_mask * token_type_ids
        else:
            a_head_mask = None
            b_head_mask = None

        if inputs_embeds is not None:
            a_inputs_embeds = inputs_embeds * (1 - token_type_ids)
            b_inputs_embeds = inputs_embeds * token_type_ids
        else:
            a_inputs_embeds = None
            b_inputs_embeds = None

        # Force zero token type ids for both outputs
        zero_token_type_ids = token_type_ids.zero_()

        a = {
            'input_ids': a_input_ids,
            'attention_mask': a_attention_mask,
            'position_ids': a_position_ids,
            'head_mask': a_head_mask,
            'inputs_embeds': a_inputs_embeds,
            'token_type_ids': zero_token_type_ids
        }

        b = {
            'input_ids': b_input_ids,
            'attention_mask': b_attention_mask,
            'position_ids': b_position_ids,
            'head_mask': b_head_mask,
            'inputs_embeds': b_inputs_embeds,
            'token_type_ids': zero_token_type_ids
        }

        return a, b

    def _pack(self, a_outputs, b_outputs):
        outputs = (
            self.combine_fn(a_outputs[0], b_outputs[0]), # Sequence output
            self.combine_fn(a_outputs[1], b_outputs[1]), # Pooled output
            (a_outputs[2:], b_outputs[2:])  # Rest
        )
        return outputs


@add_start_docstrings(
    """Seq2mat Model with a `language modeling` head on top. """, SEQ2MAT_START_DOCSTRING, SEQ2MAT_INPUTS_DOCSTRING
)
class Seq2matForMaskedLM(Seq2matPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = Seq2matTokenizer.from_pretrained('seq2mat-base-uncased')
        model = Seq2matForMaskedLM.from_pretrained('seq2mat-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)

        self.d = config.root_mat_size
        self.transformer = Seq2matModel(config)

        self.jumping_knowledge = config.jumping_knowledge
        if self.jumping_knowledge:
            hidden_size = (config.conv_layers + 1) * config.hidden_size
        else:
            hidden_size = config.hidden_size
        self.mode = config.mode # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token. (if `tied_weights`)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)
        # Need a link between the two variables so that the bias is correctly
        # resized with `resize_token_embeddings`
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.lm_head.bias = self.bias
        self.tied_weights = config.tied_weights
        self.bidirectional = config.bidirectional
        self.init_weights()

    def get_output_embeddings(self):
        # return self.lm_head
        # Returning 'None' will *not* tie weights
        if self.tied_weights:
            return self.lm_head
        else:
            return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # No more switch-case statements, as all variants now output flat vectors
        lm_inputs = outputs[0]


        # if self.mode == 'hybrid':
        #     cmow_outputs = flatten_generic(outputs[0], -3 if self.bidirectional else -2)
        #     cbow_outputs = outputs[1]
        #     if self.bidirectional:
        #         # Flatten bidirection dim
        #         cbow_outputs = flatten_generic(cbow_outputs, -2)
        #     lm_inputs = torch.cat([cmow_outputs, cbow_outputs], dim=-1)
        # elif self.mode == 'cmow':
        #     sequence_output = outputs[0]
        #     lm_inputs = flatten_generic(sequence_output, -3 if self.bidirectional else -2)
        # elif self.mode == 'recursive':
        #     sequence_output = outputs[0]
        #     lm_inputs = sequence_output
        # elif self.mode == 'conv':
        #     if self.jumping_knowledge:
        #         # If jumping knowledge, concat all hidden states across layers
        #         all_hidden_states = outputs[2]
        #         h = torch.cat(all_hidden_states, dim=-1)
        #         lm_inputs = flatten(h)
        #     else:
        #         # Otherwise: simply flatten position-wise outputs
        #         sequence_output = outputs[0]
        #         lm_inputs = flatten(sequence_output)

        prediction_scores = self.lm_head(lm_inputs)
        # print("Pred score size", prediction_scores.size())
        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        outputs = (prediction_scores, lm_inputs)
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)



@add_start_docstrings(
    """Seq2mat Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    SEQ2MAT_START_DOCSTRING,
    SEQ2MAT_INPUTS_DOCSTRING,
)
class Seq2matForSequenceClassification(Seq2matPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = Seq2matTokenizer.from_pretrained('seq2mat-base-uncased')
        model = Seq2matForSequenceClassification.from_pretrained('seq2mat-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.mode = config.mode
        self.num_labels = config.num_labels
        self.siamese = config.siamese

        # Keep it visible in this module to load old checkpoints
        self.transformer = Seq2matModel(config)

        if config.num_input_sequences > 1 and self.siamese:
            combine_fn = config.siamese_regression if config.num_labels == 1 else config.siamese
            self.siamese_model = Seq2matSiamese(self.transformer, combine_fn)
        else:
            self.siamese_model = None

        self.dropout = nn.Dropout(config.hidn_pdrop)
        # if config.layernorm:
        #     self.layernorm = nn.LayerNorm(config.hidden_size)
        # else:
        #     self.layernorm = None
        self.jumping_knowledge = config.jumping_knowledge
        if self.jumping_knowledge:
            hidden_size = (config.conv_layers + 1) * config.hidden_size
        else:
            hidden_size = config.hidden_size

        if config.mode == 'hybrid' and config.bidirectional:
            # Special treatment for this case since we only need
            # the pooled CBOW embeddings once (forward and backward pass are equal)
            # N ^ 2 + emb_dim
            hidden_size = 2 * config.root_mat_size * config.root_mat_size + config.embedding_size


        if config.num_input_sequences > 1:
            # Adjust hidden_size used for output layer if necessary
            if config.siamese == "concat":
                hidden_size = hidden_size * config.num_input_sequences
            elif config.siamese == "diffcat":
                if config.num_labels > 1:
                    # Classifiation
                    assert config.num_input_sequences == 2, "Diff+concat siamese strategy can only deal with 2 sequences"
                    # Concat two sequences along with absolute difference between the two
                    hidden_size = hidden_size * 3
                else: 
                    hidden_size = hidden_size * 3



        if config.num_labels == 1 and config.siamese_regression == "cosine":
            # In regression with cosine similarity case, we don't need a classifier
            self.classifier = None
        elif config.text_classifier == 'linear':
            self.classifier = nn.Linear(hidden_size, self.config.num_labels)
        elif config.text_classifier == 'mlp':
            # Use d_intermediate if available else default to hidden size
            text_clf_hidden_size = config.d_intermediate if config.d_intermediate is not None else hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, text_clf_hidden_size),
                nn.ReLU(),
                nn.Linear(text_clf_hidden_size, text_clf_hidden_size),
                nn.Dropout(config.hidn_pdrop),
                Seq2matLayerNorm(text_clf_hidden_size),
                nn.Linear(text_clf_hidden_size, self.config.num_labels)
            )
        else:
            raise NotImplementedError("Unknown text classifier: " + config.text_classifier)

        self.d = config.root_mat_size
        self.bidirectional = config.bidirectional

        self.init_weights()


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

        if self.jumping_knowledge:
            # Aggregate over layers for final prediction
            all_hidden_states = outputs[2] # tuple of hidden states
            num_layers = len(all_hidden_states)
            h = torch.stack(all_hidden_states, dim=-1)
            # bsz x seqlen x d x d x layers
            pooled_output = h.mean(dim=1) # is already flat
            pooled_output = h.view(-1, self.d * self.d * num_layers)
        else:
            pooled_output = outputs[1]

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
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Seq2mat Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    SEQ2MAT_START_DOCSTRING,
    SEQ2MAT_INPUTS_DOCSTRING,
)
class Seq2matForTokenClassification(Seq2matPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = Seq2matTokenizer.from_pretrained('seq2mat-base-uncased')
        model = Seq2matForTokenClassification.from_pretrained('seq2mat-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = Seq2matModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

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

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Seq2mat Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    SEQ2MAT_START_DOCSTRING,
    SEQ2MAT_INPUTS_DOCSTRING,
)
class Seq2matForQuestionAnswering(Seq2matPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = Seq2matTokenizer.from_pretrained('seq2mat-base-uncased')
        model = Seq2matForQuestionAnswering.from_pretrained('seq2mat-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
        # a nice puppet


    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = Seq2matModel(config)
        # 2 outputs: one for start logit, one for end logit
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.mode = config.mode

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        if self.mode == 'conv':
            qa_inputs = flatten(outputs[0])
        elif self.mode == 'cmow':
            qa_inputs = flatten(outputs[0])
        elif self.mode == 'hybrid':
            cmow_outputs = flatten(outputs[0])
            cbow_outputs = outputs[1]
            qa_inputs = torch.cat([cmow_outputs, cbow_outputs], dim=-1)

        logits = self.qa_outputs(qa_inputs)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)  #  + outputs[2:]
        # Outputs[2:] are not what is
        # expected in run_squad
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
