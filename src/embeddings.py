# Embeddings for SequenceTagger model.
# This includes a custom version for BPEmb which averages over tokens 
# and our Meta-Embeddings module with feature-based attention.
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import flair
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, DocumentEmbeddings
from flair.data import Dictionary, Token, Sentence
from flair.file_utils import cached_path
from flair.training_utils import log_line

import os
import re
import logging
from abc import abstractmethod
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Dict, Tuple

import hashlib

import gensim
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from bpemb import BPEmb
except:
    print('Cannot import BPEmb')
    
try:
    from transformers import AutoTokenizer, AutoConfig, AutoModel
except:
    print('Cannot import Transformers')

import torch.nn.functional as F
from torch.nn import ParameterList, Parameter
from torch.nn import Sequential, Linear, Conv2d, ReLU, MaxPool2d, Dropout2d
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from src.models import EmbeddingAttention, FeatureEmbedding


log = logging.getLogger("flair")



class AveragingBytePairEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        language: str,
        dim: int = 100,
        syllables: int = 200000,
        cache_dir=Path(flair.cache_root) / "embeddings",
        emb_method='avg',
    ):
        """
        Initializes BP embeddings. Constructor downloads required files if not there.
        """

        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.name: str = f"bpe-{language}-{syllables}-{dim}"
        self.static_embeddings = True
        self.embedder = BPEmb(lang=language, vs=syllables, dim=dim, cache_dir=cache_dir)

        self.emb_method = emb_method
        if self.emb_method in ['avg', 'first', 'last']:
            self.__embedding_length: int = self.embedder.emb.vector_size
        else:
            self.__embedding_length: int = self.embedder.emb.vector_size * 2
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word.strip() == "":
                    # empty words get no embedding
                    token.set_embedding(
                        self.name, torch.zeros(self.embedding_length, dtype=torch.float)
                    )
                else:
                    # all other words get embedded
                    embeddings = self.embedder.embed(word.lower())
                    if self.emb_method == 'first':
                        embedding = embeddings[0]
                    elif self.emb_method == 'last':
                        embedding = embeddings[-1]
                    elif self.emb_method == 'avg':
                        embedding = np.average(embeddings, axis=0)
                    else:
                        embedding = np.concatenate((embeddings[0], embeddings[-1]))
                    token.set_embedding(
                        self.name, torch.tensor(embedding, dtype=torch.float)
                    )

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return "model={}".format(self.name)

        
class MetaEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, 
                 embeddings: List[TokenEmbeddings],
                 use_average: bool = False,
                 use_attention: bool = False,
                 use_features: bool = False,
                 feature_model: FeatureEmbedding = None,
                 use_mapping_bias: bool = True,
                 use_fixed_weights_for_att: Union[bool, List[int]] = False,
                 att_hidden_size: int = 10,
                 max_mapping_dim: int = -1, 
                 use_batch_norm: bool = True,
                 use_mapping_norm: bool = False):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings
        self.num_embeddings = len(embeddings)
        self.embedding_names = []

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            embedding.name = f"{str(i)}-{embedding.name}"
            self.embedding_names.append(embedding.name)
            self.add_module(f"list_embedding_{str(i)}", embedding)

        self.name: str = "Meta"
        self.use_average = use_average
        self.use_attention = use_attention
        self.use_features = use_features
        self.static_embeddings: bool = not self.use_average

        self.__embedding_type: str = embeddings[0].embedding_type

        self.map_all_embeddings = False
        if max_mapping_dim > 0:
            self.__embedding_length = max_mapping_dim
            if not use_average:
                self.map_all_embeddings = True # If concatenated embeddings are mapped to a smaller size
                
        elif self.use_average: 
            self.__embedding_length = max([emb.embedding_length for emb in self.embeddings])
        else:
            self.__embedding_length = sum([emb.embedding_length for emb in self.embeddings])
            
        # Add Meta-Embedding models
        if self.use_average:
            self.embedding_mappings = []
            for emb in self.embeddings:
                inp_dim = emb.embedding_length
                out_dim = self.embedding_length
                l = torch.nn.Linear(inp_dim, out_dim, bias=use_mapping_bias)
                l.to(flair.device)
                self.embedding_mappings.append(l)
            self.embedding_mappings = torch.nn.ModuleList(self.embedding_mappings)
        elif self.map_all_embeddings:
            inp_dim = sum([emb.embedding_length for emb in self.embeddings])
            self.final_embedding_mapping = torch.nn.Linear(inp_dim, self.embedding_length, 
                                                           bias=use_mapping_bias).to(flair.device)
            
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(self.embedding_length)
            
            
        self.use_mapping_norm = use_mapping_norm
        if use_mapping_norm: 
            self.embedding_norms = []
            for i in range(len(self.embeddings)):
                out_dim = self.embedding_length
                l = torch.nn.LayerNorm(out_dim)
                l.to(flair.device)
                self.embedding_norms.append(l)
            self.embedding_norms = torch.nn.ModuleList(self.embedding_norms)
            
        if self.use_features:
            self.feature_model = feature_model
            self.attention = EmbeddingAttention(self.embedding_length, feature_model.embedding_length, att_hidden_size, 
                                                self.use_average, use_fixed_weights_for_att)
            
        elif self.use_attention:
            self.attention = EmbeddingAttention(self.embedding_length, 0, att_hidden_size, 
                                                self.use_average, use_fixed_weights_for_att)
            
            
        self.to(flair.device)
            
        log_line(log)
        log.info(f'Meta-Embedding Configuration')
        log_line(log)
        log.info("Embeddings:")
        for emb in self.embeddings:
            log.info(f' - {emb.name} ({emb.embedding_length})')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - static_embeddings: "{self.static_embeddings}"')
        log.info(f' - embedding_length: "{self.embedding_length}"')
        log.info(f' - use_average: "{self.use_average}"')
        log.info(f' - use_attention: "{self.use_attention}"')
        log.info(f' - use_features: "{self.use_features}"')
        log.info(f' - use_mapping_bias: "{use_mapping_bias}"')
        log.info(f' - att_hidden_size: "{att_hidden_size}"')
        log.info(f' - use_fixed_weights_for_att: "{use_fixed_weights_for_att}"')
        log.info(f' - use_batch_norm: "{use_batch_norm}"')
        log.info(f' - use_mapping_norm: "{use_mapping_norm}"')
        log.info(f' - map_all_embeddings: "{self.map_all_embeddings}"')
        log_line(log)
        
    def forward(self, 
                sentences: Union[Sentence, List[Sentence]],
                return_mapped_embeddings: bool = False,
                return_weights: bool = False):
        #self.embed(sentences)
        
        if type(sentences) is Sentence:
            sentences = [sentences]

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        if self.use_average:
            emb_size = self.embedding_length * longest_token_sequence_in_batch * self.num_embeddings
        else:
            emb_size = self.embedding_length * longest_token_sequence_in_batch
        pre_allocated_zero_tensor = torch.zeros(
            emb_size,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs: List[torch.Tensor] = list()
        for sentence in sentences:
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)
            
            if self.use_average:
                for token in sentence:
                    for i, emb in enumerate(token.get_each_embedding(self.embedding_names)):
                        all_embs.append(self.embedding_mappings[i](emb))
                    #all_embs.append(token_embs)   
                    
                if nb_padding_tokens > 0:
                    t = pre_allocated_zero_tensor[
                        : self.embedding_length * nb_padding_tokens * self.num_embeddings
                        ]
                    all_embs.append(t)
                    
            elif self.map_all_embeddings:  # for concatenation with mapping to smaller size
                for token in sentence:
                    #torch.zeros(self.embedding_length * self.num_embeddings)
                    token_embs: List[torch.Tensor] = list()
                    for i, emb in enumerate(token.get_each_embedding(self.embedding_names)):
                        token_embs.append(emb)
                    token_embs = torch.cat(token_embs)
                    token_embs = self.final_embedding_mapping(token_embs)
                    all_embs.append(token_embs)   
                    
                if nb_padding_tokens > 0:
                    t = pre_allocated_zero_tensor[
                        : self.embedding_length * nb_padding_tokens
                        ]
                    all_embs.append(t)
            
            else: # for concatenation
                all_embs += [
                    emb for token in sentence for emb in token.get_each_embedding(self.embedding_names)
                ]
                
                if nb_padding_tokens > 0:
                    t = pre_allocated_zero_tensor[
                        : self.embedding_length * nb_padding_tokens
                        ]
                    all_embs.append(t)

        if self.use_average:
            att_inp = torch.cat(all_embs).view(
                [
                    len(sentences),
                    longest_token_sequence_in_batch,
                    self.num_embeddings,
                    self.embedding_length,
                ]
            )
            
            if return_mapped_embeddings:
                mapped_embeddings = [t.squeeze(2) for t in torch.chunk(att_inp, self.num_embeddings, dim=2)]
                return mapped_embeddings
            
            if self.use_mapping_norm:
                mapped = [t.squeeze(2) for t in torch.chunk(att_inp, self.num_embeddings, dim=2)]
                normalized = [self.embedding_norms[i](emb) for i, emb in enumerate(mapped)]
                att_inp = torch.stack(normalized, dim=2)
            
            ## apply attention
            if self.use_features:
                feature_tensor = self.feature_model(sentences, (len(sentences), longest_token_sequence_in_batch))
                feature_tensor = feature_tensor.transpose_(0, 1)
                feature_inp = torch.stack([feature_tensor for x in range(self.num_embeddings)], dim=2)
                sentence_tensor, weights = self.attention(att_inp, feature_inp)
            elif self.use_attention:
                sentence_tensor, weights = self.attention(att_inp, None)
            else:
                sentence_tensor = torch.sum(att_inp, dim=2)
                
            if return_weights:
                return weights
            
            
            
        else:    
            sentence_tensor = torch.cat(all_embs).view(
                [
                    len(sentences),
                    longest_token_sequence_in_batch,
                    self.embedding_length,
                ]
            )
            
        if self.use_batch_norm and len(sentences) > 1:
            sentence_tensor = sentence_tensor.view(len(sentences) * longest_token_sequence_in_batch, self.embedding_length)
            sentence_tensor = self.batch_norm(sentence_tensor)
            sentence_tensor = sentence_tensor.view(len(sentences), longest_token_sequence_in_batch, self.embedding_length)
            
        return sentence_tensor

    def embed(
            self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)
            
        self._add_embeddings_internal(sentences)

    def embed_internal_embeddings(
            self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)
            
        sentence_tensor = self.forward(sentences)
        for sid, sent in enumerate(sentences):
            for tid, token in enumerate(sent):
                token.set_embedding(self.name, sentence_tensor[sid][tid])

        return sentences
        
    def get_names(self) -> List[str]:
        """Returns a list of embedding names. In most cases, it is just a list with one item, namely the name of
        this embedding. But in some cases, the embedding is made up by different embeddings (StackedEmbedding).
        Then, the list contains the names of all embeddings in the stack."""
        return [self.name]

    def __str__(self):
        return f'MetaEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

    def get_named_embeddings_dict(self) -> Dict:
        named_embeddings_dict = {}
        for embedding in self.embeddings:
            named_embeddings_dict.update(embedding.get_named_embeddings_dict())

        return named_embeddings_dict

    
class DocumentRNNEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            embeddings: List[TokenEmbeddings],
            hidden_size=128,
            rnn_layers=1,
            reproject_words: bool = False,
            reproject_words_dimension: int = None,
            bidirectional: bool = False,
            dropout: float = 0.5,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            rnn_type="GRU",
            fine_tune: bool = True,
    ):
        """The constructor takes a list of embeddings to be combined.
        We changed the embeddings to support our MetaEmbeddings.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        :param rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()

        #self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.embeddings = embeddings

        self.rnn_type = rnn_type

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False if fine_tune else True

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )

        # bidirectional RNN on top of embedding layer
        if rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = torch.nn.GRU(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )

        self.name = "document_" + self.rnn._get_name()

        # dropouts
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = (
            LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        )
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        # TODO: remove in future versions
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        if type(sentences) is Sentence:
            sentences = [sentences]

        #self.rnn.zero_grad()

        # embed words in the sentence
        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)
            
        self.embeddings.embed_internal_embeddings(sentences)
        sentence_tensor = self.embeddings(sentences)

        # before-RNN dropout
        if self.dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        # reproject if set
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        # push through RNN
        packed = pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )
        rnn_out, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(rnn_out, batch_first=True)

        # after-RNN dropout
        if self.dropout:
            outputs = self.dropout(outputs)
        if self.locked_dropout:
            outputs = self.locked_dropout(outputs)

        # extract embeddings from RNN
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[sentence_no, length - 1]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[sentence_no, 0]
                embedding = torch.cat([first_rep, last_rep], 0)

            if self.static_embeddings:
                embedding = embedding.detach()

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _apply(self, fn):

        # models that were serialized using torch versions older than 1.4.0 lack the _flat_weights_names attribute
        # check if this is the case and if so, set it
        for child_module in self.children():
            if isinstance(child_module, torch.nn.RNNBase) and not hasattr(child_module, "_flat_weights_names"):
                _flat_weights_names = []

                if child_module.__dict__["bidirectional"]:
                    num_direction = 2
                else:
                    num_direction = 1
                for layer in range(child_module.__dict__["num_layers"]):
                    for direction in range(num_direction):
                        suffix = "_reverse" if direction == 1 else ""
                        param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                        if child_module.__dict__["bias"]:
                            param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                        param_names = [
                            x.format(layer, suffix) for x in param_names
                        ]
                        _flat_weights_names.extend(param_names)

                setattr(child_module, "_flat_weights_names",
                        _flat_weights_names)

            child_module._apply(fn)