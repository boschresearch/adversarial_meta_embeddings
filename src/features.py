# Word-features for attention-based meta-embeddings
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

from collections import Counter, defaultdict
from string import punctuation
from tqdm.auto import tqdm
from typing import List
import os

from flair.data import Label, Token, Sentence

NUM_LENGTH_GROUPS = 20
NUM_FREQ_GROUPS = 20


def _get_shape(text: str):
    shape = ''
    for char in text:
        if char.isdigit():
            shape += 'n'
        elif char.isalpha() and char.islower():
            shape += 'c'
        elif char.isalpha() and char.isupper():
            shape += 'C'
        elif char in punctuation:
            shape += 'p'
        else:
            shape += 'u'
    return shape

def _frequencies_for_word2vec(filename):
    freq_counter = Counter()
    with open(filename, 'r') as fin:
        offset = 0
        for rank, line in enumerate(fin):
            line = line.strip()
            word = line.split(' ')[0]
            if rank > 0:
                if word in freq_counter:
                    offset -= 1
                elif word[0] == '<' and word[-1] == '>':
                    offset -= 1
                else:
                    freq = _freq_for_rank(rank + offset)
                    freq_counter[word] = freq
    return freq_counter


def _frequency_binning(counter, num_groups):
    groups = {0: []}
    step_size = sum(counter.values()) / num_groups
    
    cur_bin = 0
    cur_size = 0.0
    next_size = step_size
    
    common = counter.most_common()
    for w, f in common:
        cur_size += f
        groups[cur_bin].append(w)
        
        if cur_size + f >= next_size and cur_bin < num_groups - 1:
            next_size += step_size
            cur_bin += 1
            groups[cur_bin] = []
            
    groups = {g: vals for g, vals in groups.items() if len(vals) > 0}
    word2freq = defaultdict(lambda : num_groups - 1)
    for g, vals in groups.items():
        for word in vals:
            word2freq[word] = g
    return word2freq

def _freq_for_rank(rank, c=0.1):
    '''Calculates the frequency according to Zipfs Law'''
    return c / rank

def _add_features(sentences: List[Sentence], features, freq_bins, idx2shape, shape2idx, idx2word, word2idx):
    """ All-in-one function to add features according to "features" parameter
    features : a string with optional flags
      * 'f' for frequency
      * 's' for shape id
      * 'w' for word id
      * 'l' for length
      * 'b' for basic one-hot encoded features
        - includes punctuation, lowercased, etc.
    """
    for e_sid, sent in enumerate(sentences):
        for e_tid, token in enumerate(sent):
            t = token.text
            
            if 'w' in features:
                if t in word2idx:
                    t_id = word2idx[t]
                else:
                    t_id = len(idx2word) + 1
                    idx2word[t_id] = t
                    word2idx[t] = t_id
                token.add_tag('feat//word', t, 1.0)
                token.add_tag('feat//word-id', t_id, 1.0)
            
            if 's' in features:
                s = _get_shape(t)
                if s in shape2idx:
                    s_id = shape2idx[s]
                else:
                    s_id = len(idx2shape) + 1
                    idx2shape[s_id] = s
                    shape2idx[s] = s_id
                token.add_tag('feat//shape', s, 1.0)
                token.add_tag('feat//shape-id', s_id, 1.0)
            
            if 'f' in features:
                f = freq_bins[t] + 1
                token.add_tag('feat//freq', f, 1.0)
            
            if 'l' in features:
                l = max(min(len(t), NUM_LENGTH_GROUPS-1), 1)
                token.add_tag('feat//len', l, 1.0)
            
            if 'b' in features:
                c = t.isupper()
                c0 = t[0].isupper() 
                c1 = any([ti.isupper() for ti in t])
                n = t.isnumeric()
                n0 = t[0].isnumeric() 
                n1 = any([ti.isnumeric() for ti in t])
                a = t.isalnum()
                a0 = t[0].isalnum() 
                a1 = any([ti.isalnum() for ti in t])
                p = t in punctuation
                p0 = t[0] in punctuation
                p1 = any([ti in punctuation for ti in t])
                token.add_tag('feat//basic', (c, c0, c1, n, n0, n1, a, a0, a1, p, p0, p1), 1.0)
            #token.add_tag('feat/vec/basic', to_binary([c, c0, c1, n, n0, n1, a, a0, a1, p, p0, p1]), 1.0)
            
    feature_dims = {
        'f': NUM_FREQ_GROUPS if 'f' in features else 0, 
        'l': NUM_LENGTH_GROUPS if 'l' in features else 0,
        'w': len(word2idx) if 'w' in features else 0,
        's': len(shape2idx) if 's' in features else 0,
        'b': 12 if 'b' in features else 0,
    }
    return feature_dims


def add_features(corpus, use_frequencies_from_file='path/to/word2vec/file', use_length=True, 
                 use_words=True, use_shapes=True, use_basic=True):
    
    feature_flags = ''
    if use_frequencies_from_file and use_frequencies_from_file != 'path/to/word2vec/file':
        print(f'Retrieve frequencies from {use_frequencies_from_file}')
        freq_counts = _frequencies_for_word2vec(use_frequencies_from_file)
        freq_bins = _frequency_binning(freq_counts, NUM_FREQ_GROUPS)
        feature_flags += 'f'
    else:
        freq_bins = defaultdict(lambda : 1)
        
    if use_length:
        feature_flags += 'l'
    if use_words:
        feature_flags += 'w'
    if use_shapes:
        feature_flags += 's'
    if use_basic:
        feature_flags += 'b'
    print('Use features: ' + feature_flags)

    idx2shape, shape2idx, idx2word, word2idx = dict(), dict(), dict(), dict()
    feature_dims = _add_features(corpus._train, feature_flags, freq_bins, idx2shape, shape2idx, idx2word, word2idx)
    feature_dims = _add_features(corpus._dev, feature_flags, freq_bins, idx2shape, shape2idx, idx2word, word2idx)
    feature_dims = _add_features(corpus._test, feature_flags, freq_bins, idx2shape, shape2idx, idx2word, word2idx)
    return feature_flags, feature_dims, idx2shape, idx2word
