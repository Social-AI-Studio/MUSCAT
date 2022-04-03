# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertIntermediate,
    BertOutput,
    BertSelfOutput,
)
from .file_utils import cached_path

logger = logging.getLogger(__name__)


class SourceAttendsSubthread(nn.Module):
    def __init__(self, config):
        super(SourceAttendsSubthread, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, source_hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(source_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        return attention_scores


class MyBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(MyBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.src_subthread_attn = SourceAttendsSubthread(config)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, source_embedding=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Adding source attented attention score to original
        if source_embedding is not None:
            attention_scores2 = self.src_subthread_attn(hidden_states, source_embedding)
            attention_scores = attention_scores + attention_scores2
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class MyBertAttention(nn.Module):
    def __init__(self, config):
        super(MyBertAttention, self).__init__()
        # replacing BertSelfAttention() with modified MyBertSelfAttention()
        self.self = MyBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, source_embedding=None):
        self_output = self.self(input_tensor, attention_mask, source_embedding)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class MyBertLayer(nn.Module):
    def __init__(self, config):
        super(MyBertLayer, self).__init__()
        self.attention = MyBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, source_embedding=None):
        attention_output = self.attention(
            hidden_states, attention_mask, source_embedding
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MyBertEncoder(nn.Module):
    def __init__(self, config):
        super(MyBertEncoder, self).__init__()
        # layer = MyBertLayer(config)
        num_bert_layers = config.num_hidden_layers // 2
        num_mybert_layers = config.num_hidden_layers // 2
        assert num_bert_layers + num_mybert_layers == config.num_hidden_layers
        # self.layer = nn.ModuleList(
        #     [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        # )
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(num_bert_layers)]
            + [MyBertLayer(config) for _ in range(num_mybert_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        source_embedding=None,
        output_all_encoded_layers=True,
    ):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if isinstance(layer_module, BertLayer):
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                )
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_module(
                    hidden_states, attention_mask, source_embedding
                )
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ADDBertEncoder(nn.Module):
    def __init__(self, config):
        super(ADDBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(1)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MyBertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(MyBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = MyBertEncoder(config)
        self.pooler = BertPooler(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_all_encoded_layers=True,
        inputs_embeds=None,
        src_input_ids=None,
        src_token_type_ids=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )
        if src_token_type_ids is None:
            src_token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device
            )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        if src_input_ids is not None and src_token_type_ids is not None:
            src_embedding = self.embeddings(
                input_ids=src_input_ids,
                token_type_ids=src_token_type_ids,
                position_ids=position_ids,
            )
        else:
            src_embedding = None

        encoded_layers = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            source_embedding=src_embedding,
            output_all_encoded_layers=output_all_encoded_layers,
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class HierarchicalCoupledCoAttnBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=4):
        super(HierarchicalCoupledCoAttnBertForSequenceClassification, self).__init__(
            config
        )
        self.num_labels = num_labels
        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.add_bert_attention = ADDBertEncoder(config)
        self.add_bert_pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids1,
        attention_mask1,
        input_ids2,
        attention_mask2,
        input_ids3,
        attention_mask3,
        attention_mask,
        src_input_ids,
        labels=None,
    ):
        # sequence_output1, pooled_output = self.bert(input_ids1, token_type_ids1, attention_mask1, output_all_encoded_layers=False)
        sequence_output1, _ = self.bert(
            input_ids1,
            attention_mask=attention_mask1,
            output_all_encoded_layers=False,
            src_input_ids=src_input_ids,
        )
        sequence_output2, _ = self.bert(
            input_ids2,
            attention_mask=attention_mask2,
            output_all_encoded_layers=False,
            src_input_ids=src_input_ids,
        )
        sequence_output3, _ = self.bert(
            input_ids3,
            attention_mask=attention_mask3,
            output_all_encoded_layers=False,
            src_input_ids=src_input_ids,
        )

        tmp_sequence = torch.cat((sequence_output1, sequence_output2), dim=1)
        sequence_output = torch.cat((tmp_sequence, sequence_output3), dim=1)

        #'''
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(
        #     dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        add_bert_encoder = self.add_bert_attention(
            sequence_output, extended_attention_mask
        )
        add_bert_text_output_layer = add_bert_encoder[-1][
            0
        ]  # double check pls, could be a bug
        final_text_output = self.add_bert_pooler(add_bert_text_output_layer)
        #'''
        # final_text_output = self.add_bert_pooler(sequence_output)

        pooled_output = self.dropout(final_text_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ((loss,) + (logits,)) if loss is not None else logits


# Ablation: Coupled BERT model with CoAttention module
class CoupledCoAttnBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=4):
        super(CoupledCoAttnBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = MyBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.add_bert_pooler = BertPooler(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, num_labels),
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids1,
        attention_mask1,
        input_ids2,
        attention_mask2,
        input_ids3,
        attention_mask3,
        attention_mask,
        src_input_ids,
        labels=None,
    ):
        _, pooled_output1 = self.bert(
            input_ids1,
            attention_mask=attention_mask1,
            output_all_encoded_layers=False,
            src_input_ids=src_input_ids,
        )
        _, pooled_output2 = self.bert(
            input_ids2,
            attention_mask=attention_mask2,
            output_all_encoded_layers=False,
            src_input_ids=src_input_ids,
        )
        _, pooled_output3 = self.bert(
            input_ids3,
            attention_mask=attention_mask3,
            output_all_encoded_layers=False,
            src_input_ids=src_input_ids,
        )
        logger.debug(f"pooled --> {pooled_output1.shape}")
        tmp_pool = torch.cat((pooled_output1, pooled_output2), dim=1)
        final_pool_output = torch.cat((tmp_pool, pooled_output3), dim=1)

        pooled_output = self.dropout(final_pool_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ((loss,) + (logits,)) if loss is not None else logits
