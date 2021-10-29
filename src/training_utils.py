# Fixed WeightExtractor to handle single item tensors, e.g. batchnorm.num_batches_tracked
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
#
# This source code is based on the flairNLP Project v0.8
#   (https://github.com/flairNLP/flair/releases/tag/v0.8)
# Copyright (c) 2018 Zalando SE, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from collections import defaultdict
from pathlib import Path
from typing import Union, List
import random

import flair
from flair.data import Dictionary, Sentence
from flair.training_utils import init_output_file
from functools import reduce

class WeightExtractor(object):
    def __init__(self, directory: Union[str, Path], number_of_weights: int = 10):
        if type(directory) is str:
            directory = Path(directory)
        self.weights_file = init_output_file(directory, "weights.txt")
        self.weights_dict = defaultdict(lambda: defaultdict(lambda: list()))
        self.number_of_weights = number_of_weights

    def extract_weights(self, state_dict, iteration):
        for key in state_dict.keys():

            vec = state_dict[key]
            try:
                weights_to_watch = min(
                    self.number_of_weights, reduce(lambda x, y: x * y, list(vec.size()))
                )
            except:
                continue

            if key not in self.weights_dict:
                self._init_weights_index(key, state_dict, weights_to_watch)

            for i in range(weights_to_watch):
                vec = state_dict[key]
                for index in self.weights_dict[key][i]:
                    vec = vec[index]

                value = vec.item()

                with open(self.weights_file, "a") as f:
                    f.write("{}\t{}\t{}\t{}\n".format(iteration, key, i, float(value)))

    def _init_weights_index(self, key, state_dict, weights_to_watch):
        indices = {}

        i = 0
        while len(indices) < weights_to_watch:
            vec = state_dict[key]
            cur_indices = []

            for x in range(len(vec.size())):
                index = random.randint(0, len(vec) - 1)
                vec = vec[index]
                cur_indices.append(index)

            if cur_indices not in list(indices.values()):
                indices[i] = cur_indices
                i += 1

        self.weights_dict[key] = indices