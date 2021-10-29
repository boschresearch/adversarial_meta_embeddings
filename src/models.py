# SequenceTagger model using attention-based meta-embeddings
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

from typing import List, Dict, Optional, Union, Callable

import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch import autograd, nn
from tabulate import tabulate

import flair
import flair.nn
import flair.embeddings
from flair.data import Token, Sentence

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"

import numpy as np
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer
from flair.datasets import SentenceDataset, StringDataset, DataLoader
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path
from flair.training_utils import Metric, Result, store_embeddings
from torch.nn.parameter import Parameter
from pathlib import Path

def to_scalar(var):
    return var.view(-1).detach().tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx


def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


def pad_tensors(tensor_list):
    ml = max([x.shape[0] for x in tensor_list])
    shape = [len(tensor_list), ml] + list(tensor_list[0].shape[1:])
    template = torch.zeros(*shape, dtype=torch.long, device=flair.device)
    lens_ = [x.shape[0] for x in tensor_list]
    for i, tensor in enumerate(tensor_list):
        template[i, : lens_[i]] = tensor

    return template, lens_


##
## Discriminator for Adversarial Training
##

class GradientReverse(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


class DomainClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_domains,
                 dropout,
                 lambd,
                 nonlinear=True):
        super(DomainClassifier, self).__init__()
        self.num_domains = num_domains
        self.lambd = lambd
        self.net = nn.Sequential()
        if dropout > 0:
            self.net.add_module('q-dropout', nn.Dropout(p=dropout))
        self.net.add_module('q-linear', nn.Linear(input_size, hidden_size))
        #self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
        if nonlinear:
            self.net.add_module('q-relu', nn.ReLU())
        self.net.add_module('q-linear-final', nn.Linear(hidden_size, num_domains))
        self.net.add_module('q-logsoftmax', nn.LogSoftmax(dim=-1))
        self.to(flair.device)

    def forward(self, x):
        #x = GradReverse.apply(x, self.lambd)
        x = grad_reverse(x, self.lambd)
        scores = self.net(x)
        return scores
    
    
    
##
## Class for Feature Vector Computation
##

def one_hot_encoded(number, size):
    vec = [0]*size
    vec[number] = 1
    return vec

def to_binary(arr):
    vec = [1 if v else 0 for v in arr]
    return vec


class FeatureEmbedding(torch.nn.Module):
    """Auxiliary class used to retrieve and convert features into tensors."""
    def __init__(self, 
                 use_features, 
                 feature_dims, 
                 idx2word={}, 
                 word_dim=100,
                 idx2shape={}, 
                 shape_dim=100,
                 use_dense=True):
        super(FeatureEmbedding, self).__init__()
        self.use_words = 'w' in use_features
        self.use_basics = 'b' in use_features
        self.use_shapes = 's' in use_features
        self.use_lengths = 'l' in use_features
        self.use_frequencies = 'f' in use_features
        self.feature_dims = feature_dims
        self.shape_dim = shape_dim
        self.word_dim = word_dim
        
        if self.use_shapes:
            # shape_embedding[0] is reserved for padding values
            self.shape_embedding = torch.nn.Embedding(len(idx2shape)+1, shape_dim)
            
        if self.use_words:
            # shape_embedding[0] is reserved for padding values
            self.word_embedding = torch.nn.Embedding(len(idx2word)+1, word_dim)
        
        self.embedding_length = self.get_embedding_length()
        self.use_dense = use_dense
        if self.use_dense and self.embedding_length > 0:
            self.feat_dense = torch.nn.Linear(self.embedding_length, self.embedding_length)
            self.activation = torch.nn.ReLU()
        
        self.to(flair.device)
            
            
    def get_embedding_length(self, include_shape=True):
        length = 0
        if self.use_basics:
            length += self.feature_dims['b']
        if self.use_lengths:
            length += self.feature_dims['l']
        if self.use_frequencies:
            length += self.feature_dims['f']
        if self.use_shapes and include_shape:
            length += self.shape_dim
        if self.use_words and include_shape:
            length += self.word_dim
        return length
    
            
    def forward(self, sentences: List[Sentence], shape):
        batch_size, seq_len = shape
        
        feature_tensor = torch.zeros(batch_size, seq_len, self.get_embedding_length(include_shape=False), device=flair.device)
        shape_ids = torch.zeros([batch_size, seq_len], dtype=torch.long, device=flair.device)
        word_ids = torch.zeros([batch_size, seq_len], dtype=torch.long, device=flair.device)
        
        for s, sent in enumerate(sentences):
            for t, token in enumerate(sent):
                vec = []
                if self.use_basics:
                    vec.extend(to_binary(token.get_tag('feat//basic').value))
                if self.use_lengths:
                    vec.extend(one_hot_encoded(int(token.get_tag('feat//len').value)-1, self.feature_dims['l']))
                if self.use_frequencies:
                    vec.extend(one_hot_encoded(token.get_tag('feat//freq').value-1, self.feature_dims['f']))
                if self.use_words:
                    word_id = token.get_tag('feat//word-id').value
                    word_ids[s][t] = word_id
                if self.use_shapes:
                    shape_id = token.get_tag('feat//shape-id').value
                    shape_ids[s][t] = shape_id
                    #vec.extend(self.shape_embedding(torch.tensor(shape_id).to(flair.device)))
                    # we might want to change this for a single access to shape_embedding per batch
                feature_tensor[s][t] = torch.tensor(vec, device=flair.device)
                
        feature_tensor = feature_tensor.to(flair.device)
             
        if self.use_shapes:
            shape_tensor = self.shape_embedding(shape_ids)
            feature_tensor = torch.cat((feature_tensor, shape_tensor), 2)
             
        if self.use_words:
            word_tensor = self.word_embedding(word_ids)
            feature_tensor = torch.cat((feature_tensor, word_tensor), 2)
            
        if self.use_dense:
            feature_tensor = self.activation(self.feat_dense(feature_tensor))
            
        return feature_tensor
   

    
##
## Attention Models for Embedding Weighting
##
class EmbeddingAttention(torch.nn.Module): 
    """
    An embedding-augmented attention layer where the attention weight is
    a = V . tanh(Ux + Wf)
    where x is the input embedding, and f is a word feature vector. 
    """
    
    def __init__(self, input_size, feature_size, attn_size, use_att_sum, fixed_weights=None):
        super(EmbeddingAttention, self).__init__()
        self.use_att_sum = use_att_sum
        self.input_size = input_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        
        self.fixed_weights = fixed_weights
        if fixed_weights is False:
            self.fixed_weights = None
        
        if self.fixed_weights is None:
            self.ulinear = torch.nn.Linear(input_size, attn_size)
            if self.feature_size > 0:
                self.wlinear = torch.nn.Linear(feature_size, attn_size)
            self.tlinear = torch.nn.Linear(attn_size, 1)
            self.init_weights()
            print(f'Initialized Attention with size ({input_size, feature_size, attn_size})')
        else:
            print(f'Use fixed weights for Attention: {self.fixed_weights}')
        self.to(flair.device)

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        if self.feature_size > 0:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning
        
    def forward(self, x, f):
        """
        x : batch_size * seq_len * emb_num * input_size
        f : batch_size * seq_len * emb_num * feature_size
        """
        seq_len, batch_size, emb_num, input_size = x.size()
        
        if self.fixed_weights is not None:
            weights = torch.tensor(self.fixed_weights).to(flair.device)
            weights = weights.repeat(seq_len * batch_size)
            weights = weights.view(seq_len * batch_size, emb_num)
            
        else:
            x_proj = self.ulinear(x.contiguous().view(-1, self.input_size))
            x_proj = x_proj.view(seq_len, batch_size, emb_num, self.attn_size)
        
            if self.feature_size > 0:
                f_proj = self.wlinear(f.contiguous().view(-1, self.feature_size))
                f_proj = f_proj.view(seq_len, batch_size, emb_num, self.attn_size)
                projs = [x_proj, f_proj]
            else:
                projs = [x_proj]
                
            scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(seq_len, batch_size, emb_num)
            weights = F.softmax(scores, dim=2)
            # weighted average input vectors
            weights = weights.view(seq_len * batch_size, emb_num)
            
        x = x.view(seq_len * batch_size, emb_num, input_size)
        return_weights = weights
        
        if self.use_att_sum:
            outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
            outputs = outputs.view(seq_len, batch_size, input_size)
        else:
            weights = weights.view(seq_len, batch_size, emb_num, 1)
            weights = weights.expand(-1, -1, -1, input_size)
            weights = weights.contiguous().view(batch_size * seq_len, emb_num, input_size)
            outputs = weights * x
            outputs = outputs.view(seq_len, batch_size, emb_num * input_size)
        
        return outputs, return_weights
    
    