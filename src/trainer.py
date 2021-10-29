# Trainer class for adversarial/multitask training
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

import copy
import logging
from pathlib import Path
from typing import List, Union
import os
import time
import datetime
import random
import sys
import inspect

import torch
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

try:
    from apex import amp
except ImportError:
    amp = None

import flair
import flair.nn
from flair.data import MultiCorpus, Corpus
from flair.datasets import DataLoader
from flair.models import SequenceTagger
from flair.optim import ExpAnnealLR
from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    log_line,
    add_file_handler,
    Result,
    store_embeddings,
    AnnealOnPlateau,
)

import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from src.training_utils import WeightExtractor

log = logging.getLogger("flair")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns
sns.set_palette('muted')
plt.style.use('seaborn-paper')

def plot_sample(E, data, method='pca', RS=548, save_as=None):
    Y, X, words = [], [], []
    
    for s, sentence in enumerate(data):
        E.embed(sentence)
        embeddings = E(sentence, return_mapped_embeddings=True)
        
        for emb_id, emb in enumerate(embeddings):
            for tid, token in enumerate(sentence):
                x = emb[0][tid]
                X.append(x.cpu().detach().numpy())
                Y.append(emb_id)
                words.append(token.text)
    X, Y = np.array(X), np.array(Y)
    
    # Apply pca and t-sne to X
    if method.lower() == 'pca':
        pca = PCA(n_components=2, random_state=RS).fit(X)  
        X_pca = pca.transform(X)
        x = X_pca
        
    elif method.lower().replace('-', '') == 'tsne':
        pca = PCA(n_components=50, random_state=RS).fit(X) 
        X_pca = pca.transform(X)
        X_tsne = TSNE(random_state=RS).fit_transform(X_pca)
        x = X_tsne
    
    np.savetxt(f'{save_as}_data_y.np', Y)
    np.savetxt(f'{save_as}_data_x.np', x)
        
    label_X = 'PC 1' if method.lower() == 'pca' else 't-SNE 1'
    label_Y = 'PC 2' if method.lower() == 'pca' else 't-SNE 2'
    
    # Create Plot
    colors = Y
    axis_off = False
    
    palette = np.array(sns.color_palette("hls", E.num_embeddings))
    num_classes = len(np.unique(colors))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    
    #if title is not None:
    #    plt.title(title)
    plt.xlabel(label_X)
    plt.ylabel(label_Y)
    
    if axis_off:
        ax.axis('off')
    ax.axis('tight')
    
    if save_as is not None:
        plt.savefig(f'{save_as}.pdf', bbox_inches='tight')
        plt.savefig(f'{save_as}.png', bbox_inches='tight')
    plt.show()
    


def get_mask(org, maxlen):
    mask = []
    for val in org:
        for i in range(maxlen):
            mask.append(1 if i < val else 0)
    mask = torch.tensor(mask)
    return mask.to(flair.device)


class AdversarialModelTrainer:
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        optimizer: torch.optim.Optimizer = SGD,
        epoch: int = 0,
        use_tensorboard: bool = False,
        D = None, 
    ):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model
        :param corpus: The dataset used to train the model, should be of type Corpus
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
        :param use_tensorboard: If True, writes out tensorboard information
        """
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus
        self.optimizer: torch.optim.Optimizer = optimizer
        self.epoch: int = epoch
        self.use_tensorboard: bool = use_tensorboard
            
        self.D = D                     #
        if D is not None:              # Our objects
            try:
                self.E = model.embeddings 
            except:
                self.E = model.document_embeddings.embeddings

    def train(
        self,
        base_path: Union[Path, str],
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        mini_batch_chunk_size: int = None,
        max_epochs: int = 100,
        scheduler = AnnealOnPlateau,
        cycle_momentum: bool = False,
        anneal_factor: float = 0.5,
        patience: int = 3,
        initial_extra_patience = 0,
        min_learning_rate: float = 0.0001,
        train_with_dev: bool = False,
        train_with_test: bool = False,
        monitor_train: bool = False,
        monitor_test: bool = False,
        embeddings_storage_mode: str = "cpu",
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        anneal_with_prestarts: bool = False,
        batch_growth_annealing: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        write_weights: bool = False,
        num_workers: int = 6,
        sampler=None,
        use_amp: bool = False,
        amp_opt_level: str = "O1",
        eval_on_train_fraction=0.0,
        eval_on_train_shuffle=False,
        save_model_at_each_epoch=False,
        adversarial_learning_k: int = 0,   # Our new parameters
        learning_rate_bert=0.0,            #
        given_weight_decay=0.0,            #
        use_bert_optimizer=False,          #
        create_space_plots=False,
        plot_sample_size=25, 
        plot_dir=None, 
        **kwargs,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved
        :param learning_rate: Initial learning rate (or max, if scheduler is OneCycleLR)
        :param mini_batch_size: Size of mini-batches during training
        :param mini_batch_chunk_size: If mini-batches are larger than this number, they get broken down into chunks of this size for processing purposes
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param scheduler: The learning rate scheduler to use
        :param cycle_momentum: If scheduler is OneCycleLR, whether the scheduler should cycle also the momentum
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
         until annealing the learning rate
        :param min_learning_rate: If the learning rate falls below this threshold, training terminates
        :param train_with_dev: If True, training is performed using both train+dev data
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing
        parameter selection.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.
        :param eval_on_train_fraction: the fraction of train data to do the evaluation on,
        if 0. the evaluation is not performed on fraction of training data,
        if 'dev' the size is determined from dev set size
        :param eval_on_train_shuffle: if True the train data fraction is determined on the start of training
        and kept fixed during training, otherwise it's sampled at beginning of each epoch
        :param save_model_at_each_epoch: If True, at each epoch the thus far trained model will be saved
        :param kwargs: Other arguments for the Optimizer
        :return:
        """
        
        if plot_dir is None:
            plot_dir = str(base_path)
            plot_dir = plot_dir + 'plots/' if plot_dir.endswith('/') else plot_dir + '/plots/'
        plot_dir = plot_dir if plot_dir.endswith('/') else plot_dir + '/' 
        os.makedirs(plot_dir, exist_ok=True)

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter()
            except:
                log_line(log)
                log.warning(
                    "ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
                )
                log_line(log)
                self.use_tensorboard = False
                pass

        if use_amp:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex."
                )

        if mini_batch_chunk_size is None:
            mini_batch_chunk_size = mini_batch_size
        if learning_rate < min_learning_rate:
            min_learning_rate = learning_rate / 10

        initial_learning_rate = learning_rate

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")

        #log_line(log)
        #log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log_line(log)
        log.info("Meta-Embedding Parameters:")
        log.info(f' - adversarial_learning_k: "{adversarial_learning_k}"')
        log.info(f' - learning_rate_bert: "{learning_rate_bert}"')
        log.info(f' - given_weight_decay: "{given_weight_decay}"')
        log.info(f' - use_bert_optimizer: "{use_bert_optimizer}"')
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Device: {flair.device}")
        log_line(log)
        log.info(f"Embeddings storage mode: {embeddings_storage_mode}")
        if isinstance(self.model, SequenceTagger) and self.model.weight_dict and self.model.use_crf:
            log_line(log)
            log.warning(f'WARNING: Specified class weights will not take effect when using CRF')

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = (
            True
            if (not param_selection_mode and self.corpus.test and monitor_test)
            else False
        )
        log_dev = False if train_with_dev or not self.corpus.dev else True
        log_train_part = (
            True
            if (eval_on_train_fraction == "dev" or eval_on_train_fraction > 0.0)
            else False
        )

        if log_train_part:
            train_part_size = (
                len(self.corpus.dev)
                if eval_on_train_fraction == "dev"
                else int(len(self.corpus.train) * eval_on_train_fraction)
            )
            assert train_part_size > 0
            if not eval_on_train_shuffle:
                train_part_indices = list(range(train_part_size))
                train_part = torch.utils.data.dataset.Subset(
                    self.corpus.train, train_part_indices
                )

        # prepare loss logging file and set up header
        loss_txt = init_output_file(base_path, "loss.tsv")

        weight_extractor = WeightExtractor(base_path)
        
        if use_bert_optimizer:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not "bert" in n],
                 'weight_decay': given_weight_decay, 'lr': learning_rate},
                {'params': [p for n, p in self.model.named_parameters() if "bert" in n], 
                 'weight_decay': 0.0, 'lr': learning_rate_bert},
             ]
            parameters = optimizer_grouped_parameters
            optimizer: torch.optim.Optimizer = self.optimizer(
                parameters, **kwargs
            )
        else:
            parameters = self.model.parameters()
            optimizer: torch.optim.Optimizer = self.optimizer(
                parameters, lr=learning_rate, **kwargs
            )

        if use_amp:
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=amp_opt_level
            )

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev else "max"
        
        if scheduler == OneCycleLR:
            dataset_size = len(self.corpus.train)
            if train_with_dev:
                dataset_size += len(self.corpus.dev)
            lr_scheduler = OneCycleLR(optimizer,
                                   max_lr=learning_rate,
                                   steps_per_epoch=dataset_size//mini_batch_size + 1,
                                   epochs=max_epochs-self.epoch, # if we load a checkpoint, we have already trained for self.epoch
                                   pct_start=0.0,
                                   cycle_momentum=cycle_momentum)
        else:
            lr_scheduler = scheduler(
                optimizer,
                factor=anneal_factor,
                patience=patience,
                initial_extra_patience=initial_extra_patience,
                mode=anneal_mode,
                verbose=True,
            )
        
        if (isinstance(lr_scheduler, OneCycleLR) and batch_growth_annealing):
            raise ValueError("Batch growth with OneCycle policy is not implemented.")

        train_data = self.corpus.train

        # if training also uses dev/train data, include in training set
        if train_with_dev or train_with_test:

            parts = [self.corpus.train]
            if train_with_dev: parts.append(self.corpus.dev)
            if train_with_test: parts.append(self.corpus.test)

            train_data = ConcatDataset(parts)

        # initialize sampler if provided
        if sampler is not None:
            # init with default values if only class is provided
            if inspect.isclass(sampler):
                sampler = sampler()
            # set dataset to sample from
            sampler.set_dataset(train_data)
            shuffle = False

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []

        micro_batch_size = mini_batch_chunk_size
        
        # store some statistics
        if self.D is not None:
            self.D.num_words = {e: [0 for j in range(0, self.E.num_embeddings)] for e in range(self.epoch, max_epochs + 1)}
            self.D.d_correct = {e: [0 for j in range(0, self.E.num_embeddings)] for e in range(self.epoch, max_epochs + 1)}

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate
            momentum = 0
            for group in optimizer.param_groups:
                if "momentum" in group:
                    momentum = group["momentum"]

            for self.epoch in range(self.epoch + 1, max_epochs + 1):
                log_line(log)
                
                if create_space_plots:
                    log.info("Create PCA plot")
                    plot_sample(self.E, self.corpus._train[0:plot_sample_size], method='pca',
                                save_as=f'{plot_dir}epoch_{self.epoch-1:03d}_pca')

                    log.info("Create t-SNE plot")
                    plot_sample(self.E, self.corpus._train[0:plot_sample_size], method='t-SNE',
                                save_as=f'{plot_dir}epoch_{self.epoch-1:03d}_tsne')

                if anneal_with_prestarts:
                    last_epoch_model_state_dict = copy.deepcopy(self.model.state_dict())

                if eval_on_train_shuffle:
                    train_part_indices = list(range(self.corpus.train))
                    random.shuffle(train_part_indices)
                    train_part_indices = train_part_indices[:train_part_size]
                    train_part = torch.utils.data.dataset.Subset(
                        self.corpus.train, train_part_indices
                    )

                # get new learning rate
                for group in optimizer.param_groups:
                    learning_rate = group["lr"]

                if learning_rate != previous_learning_rate and batch_growth_annealing:
                    mini_batch_size *= 2

                # reload last best model if annealing with restarts is enabled
                if (
                    (anneal_with_restarts or anneal_with_prestarts)
                    and learning_rate != previous_learning_rate
                    and (base_path / "best-model.pt").exists()
                ):
                    if anneal_with_restarts:
                        log.info("resetting to best model")
                        self.model.load_state_dict(
                            self.model.load(base_path / "best-model.pt").state_dict()
                        )
                    if anneal_with_prestarts:
                        log.info("resetting to pre-best model")
                        self.model.load_state_dict(
                            self.model.load(base_path / "pre-best-model.pt").state_dict()
                        )

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if (not isinstance(lr_scheduler, OneCycleLR)) and learning_rate < min_learning_rate:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle if self.epoch > 1 else False, # never shuffle the first epoch
                    num_workers=num_workers,
                    sampler=sampler,
                )

                self.model.train()

                train_loss: float = 0

                seen_batches = 0
                total_number_of_batches = len(batch_loader)

                modulo = max(1, int(total_number_of_batches / 10))

                # process mini-batches
                batch_time = 0
                for batch_no, batch in enumerate(batch_loader):

                    start_time = time.time()
                    
                    # Setup optimizer for adversarial training
                    do_adversarial_training = False
                    if adversarial_learning_k > 0 and batch_no % adversarial_learning_k == 0:
                        do_adversarial_training = True
                        if use_bert_optimizer:
                            optimizer_grouped_parameters = [
                                {'params': [p for n, p in self.model.named_parameters() if not "bert" in n],
                                 'weight_decay': given_weight_decay, 'lr': learning_rate * self.D.lambd},
                                {'params': [p for n, p in self.model.named_parameters() if "bert" in n], 
                                 'weight_decay': 0.0, 'lr': learning_rate_bert * self.D.lambd},
                                {'params': [p for p in self.D.parameters()],
                                 'weight_decay': given_weight_decay, 'lr': learning_rate * self.D.lambd},
                            ]
                            parameters = optimizer_grouped_parameters
                            optD = self.optimizer(parameters)
                        else:
                            parameters = list(self.E.parameters()) +\
                                         list(self.D.parameters())
                            optD = self.optimizer(parameters, lr=learning_rate * self.D.lambd)
                        optD.zero_grad()
                        self.D.zero_grad()    # Our model part

                    # zero the gradients on the model and optimizer
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    # if necessary, make batch_steps
                    batch_steps = [batch]
                    if len(batch) > micro_batch_size:
                        batch_steps = [
                            batch[x : x + micro_batch_size]
                            for x in range(0, len(batch), micro_batch_size)
                        ]

                    # forward and backward for batch
                    for batch_step in batch_steps:

                        # forward pass
                        loss = self.model.forward_loss(batch_step)
                        
                        #
                        # Our Adversarial Training
                        # <----
                        #
                        if do_adversarial_training:
                            sentences = batch
                            data = self.E(batch, return_mapped_embeddings=True) 
                            mask_data = [len(sent) for sent in sentences]
                            for j, d_inputs in enumerate(data):
                                batch_len, max_len, emb_size = d_inputs.size()
                                mask = get_mask(mask_data, max_len)
                                
                                d_targets = torch.tensor([j] * (max_len * batch_len), device=flair.device)
                                d_targets = d_targets.view(max_len * batch_len, )
                                d_targets = ((d_targets+1) * mask) -1
                                d_outputs = self.D(d_inputs).view(max_len * batch_len, self.D.num_domains)
            
                                loss_adv = F.nll_loss(d_outputs, d_targets, ignore_index=-1, size_average=True)
                                loss_adv.backward(retain_graph=True)
                                
                                _, pred = torch.max(d_outputs, 1)
                                self.D.num_words[self.epoch][j] += sum(mask_data)
                                self.D.d_correct[self.epoch][j] += (pred==d_targets).sum().item()
                        # ---->
                        #

                        # Backward
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()
                    if do_adversarial_training:
                        optD.step()  # our new optimizer
                    
                    # do the scheduler step if one-cycle
                    if isinstance(lr_scheduler, OneCycleLR):
                        lr_scheduler.step()
                        # get new learning rate
                        for group in optimizer.param_groups:
                            learning_rate = group["lr"]
                            if "momentum" in group:
                                momentum = group["momentum"]                    

                    seen_batches += 1
                    train_loss += loss.item()

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(batch, embeddings_storage_mode)

                    batch_time += time.time() - start_time
                    if seen_batches % modulo == 0:
                        momentum_info = f' - momentum: {momentum:.4f}' if cycle_momentum else ''
                        log.info(
                            f"epoch {self.epoch} - iter {seen_batches}/{total_number_of_batches} - loss "
                            f"{train_loss / seen_batches:.8f} - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
                            f" - lr: {learning_rate:.6f}{momentum_info}"
                        )
                        batch_time = 0
                        iteration = self.epoch * total_number_of_batches + batch_no
                        if not param_selection_mode and write_weights:
                            weight_extractor.extract_weights(
                                self.model.state_dict(), iteration
                            )

                train_loss /= seen_batches

                self.model.eval()

                log_line(log)
                log.info(
                    f"EPOCH {self.epoch} done: loss {train_loss:.4f} - lr {learning_rate:.7f}"
                )

                if self.use_tensorboard:
                    writer.add_scalar("train_loss", train_loss, self.epoch)

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train:
                    train_eval_result, train_loss = self.model.evaluate(
                        self.corpus.train,
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{train_eval_result.log_line}"

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.train, embeddings_storage_mode)

                if log_train_part:
                    train_part_eval_result, train_part_loss = self.model.evaluate(
                        train_part,
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += (
                        f"\t{train_part_loss}\t{train_part_eval_result.log_line}"
                    )
                    log.info(
                        f"TRAIN_SPLIT : loss {train_part_loss} - score {round(train_part_eval_result.main_score, 4)}"
                    )

                if log_dev:
                    dev_eval_result, dev_loss = self.model.evaluate(
                        self.corpus.dev,
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        out_path=base_path / "dev.tsv",
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
                    log.info(
                        f"DEV : loss {dev_loss} - score {round(dev_eval_result.main_score, 4)}"
                    )
                    # calculate scores using dev data if available
                    # append dev score to score history
                    dev_score_history.append(dev_eval_result.main_score)
                    dev_loss_history.append(dev_loss.item())

                    current_score = dev_eval_result.main_score

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.dev, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("dev_loss", dev_loss, self.epoch)
                        writer.add_scalar(
                            "dev_score", dev_eval_result.main_score, self.epoch
                        )

                if log_test:
                    test_eval_result, test_loss = self.model.evaluate(
                        self.corpus.test,
                        mini_batch_size=mini_batch_chunk_size,
                        num_workers=num_workers,
                        out_path=base_path / "test.tsv",
                        embedding_storage_mode=embeddings_storage_mode,
                    )
                    result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
                    log.info(
                        f"TEST : loss {test_loss} - score {round(test_eval_result.main_score, 4)}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.test, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("test_loss", test_loss, self.epoch)
                        writer.add_scalar(
                            "test_score", test_eval_result.main_score, self.epoch
                        )

                # determine learning rate annealing through scheduler. Use auxiliary metric for AnnealOnPlateau
                if log_dev and isinstance(lr_scheduler, AnnealOnPlateau):
                    lr_scheduler.step(current_score, dev_loss)
                elif not isinstance(lr_scheduler, OneCycleLR):
                    lr_scheduler.step(current_score)

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = lr_scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    new_learning_rate = group["lr"]
                if new_learning_rate != previous_learning_rate:
                    bad_epochs = patience + 1
                    if previous_learning_rate == initial_learning_rate: bad_epochs += initial_extra_patience

                # log bad epochs
                log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

                # output log file
                with open(loss_txt, "a") as f:

                    # make headers on first epoch
                    if self.epoch == 1:
                        f.write(
                            f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
                        )

                        if log_train:
                            f.write(
                                "\tTRAIN_"
                                + "\tTRAIN_".join(
                                    train_eval_result.log_header.split("\t")
                                )
                            )
                        if log_train_part:
                            f.write(
                                "\tTRAIN_PART_LOSS\tTRAIN_PART_"
                                + "\tTRAIN_PART_".join(
                                    train_part_eval_result.log_header.split("\t")
                                )
                            )
                        if log_dev:
                            f.write(
                                "\tDEV_LOSS\tDEV_"
                                + "\tDEV_".join(dev_eval_result.log_header.split("\t"))
                            )
                        if log_test:
                            f.write(
                                "\tTEST_LOSS\tTEST_"
                                + "\tTEST_".join(
                                    test_eval_result.log_header.split("\t")
                                )
                            )

                    f.write(
                        f"\n{self.epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
                    )
                    f.write(result_line)

                # if checkpoint is enabled, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.save_checkpoint(base_path / "checkpoint.pt")

                # if we use dev data, remember best model based on dev evaluation score
                if (
                    (not train_with_dev or anneal_with_restarts or anneal_with_prestarts)
                    and not param_selection_mode
                    and not isinstance(lr_scheduler, OneCycleLR)
                    and current_score == lr_scheduler.best
                    and bad_epochs == 0
                ):
                    print("saving best model")
                    self.model.save(base_path / "best-model.pt")

                    if anneal_with_prestarts:
                        current_state_dict = self.model.state_dict()
                        self.model.load_state_dict(last_epoch_model_state_dict)
                        self.model.save(base_path / "pre-best-model.pt")
                        self.model.load_state_dict(current_state_dict)
                        
                if save_model_at_each_epoch:
                    print("saving model of current epoch")
                    model_name = "model_epoch_" + str(self.epoch) + ".pt"
                    self.model.save(base_path / model_name)

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt")

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if self.use_tensorboard:
                writer.close()

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt")
                log.info("Done.")

        # test best model if test data is present
        if self.corpus.test and not train_with_test:
            final_score = self.final_test(base_path, mini_batch_chunk_size, num_workers)
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(log_handler)

        if self.use_tensorboard:
            writer.close()

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
        }

    def save_checkpoint(self, model_file: Union[str, Path]):
        corpus = self.corpus
        self.corpus = None
        torch.save(self, str(model_file), pickle_protocol=4)
        self.corpus = corpus

    @classmethod
    def load_checkpoint(cls, checkpoint: Union[Path, str], corpus: Corpus):
        model: ModelTrainer = torch.load(checkpoint, map_location=flair.device)
        model.corpus = corpus
        return model

    def final_test(
        self, base_path: Union[Path, str], eval_mini_batch_size: int, num_workers: int = 8
    ):
        if type(base_path) is str:
            base_path = Path(base_path)

        log_line(log)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")

        test_results, test_loss = self.model.evaluate(
            self.corpus.test,
            mini_batch_size=eval_mini_batch_size,
            num_workers=num_workers,
            out_path=base_path / "test.tsv",
            embedding_storage_mode="none",
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                if subcorpus.test:
                    subcorpus_results, subcorpus_loss = self.model.evaluate(
                        subcorpus.test,
                        mini_batch_size=eval_mini_batch_size,
                        num_workers=num_workers,
                        out_path=base_path / f"{subcorpus.name}-test.tsv",
                        embedding_storage_mode="none",
                    )
                    log.info(subcorpus.name)
                    log.info(subcorpus_results.log_line)

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

    def find_learning_rate(
        self,
        base_path: Union[Path, str],
        file_name: str = "learning_rate.tsv",
        start_learning_rate: float = 1e-7,
        end_learning_rate: float = 10,
        iterations: int = 100,
        mini_batch_size: int = 32,
        stop_early: bool = True,
        smoothing_factor: float = 0.98,
        **kwargs,
    ) -> Path:
        best_loss = None
        moving_avg_loss = 0

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)
        learning_rate_tsv = init_output_file(base_path, file_name)

        with open(learning_rate_tsv, "a") as f:
            f.write("ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n")

        optimizer = self.optimizer(
            self.model.parameters(), lr=start_learning_rate, **kwargs
        )

        train_data = self.corpus.train

        scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)

        model_state = self.model.state_dict()
        self.model.train()

        step = 0
        while step < iterations:
            batch_loader = DataLoader(
                train_data, batch_size=mini_batch_size, shuffle=True
            )
            for batch in batch_loader:
                step += 1

                # forward pass
                loss = self.model.forward_loss(batch)

                # update optimizer and scheduler
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                scheduler.step(step)

                print(scheduler.get_lr())
                learning_rate = scheduler.get_lr()[0]

                loss_item = loss.item()
                if step == 1:
                    best_loss = loss_item
                else:
                    if smoothing_factor > 0:
                        moving_avg_loss = (
                            smoothing_factor * moving_avg_loss
                            + (1 - smoothing_factor) * loss_item
                        )
                        loss_item = moving_avg_loss / (
                            1 - smoothing_factor ** (step + 1)
                        )
                    if loss_item < best_loss:
                        best_loss = loss

                if step > iterations:
                    break

                if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):
                    log_line(log)
                    log.info("loss diverged - stopping early!")
                    step = iterations
                    break

                with open(str(learning_rate_tsv), "a") as f:
                    f.write(
                        f"{step}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n"
                    )

            self.model.load_state_dict(model_state)
            self.model.to(flair.device)

        log_line(log)
        log.info(f"learning rate finder finished - plot {learning_rate_tsv}")
        log_line(log)

        return Path(learning_rate_tsv)
