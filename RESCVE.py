# Difference: Change RELU to GELU, add layer normalization to Transformer
# Difference 2: Add concat prediction to repr_attn, change forward to <mask> sequence
# Difference 3: remove residue layer because it does not incorporate with the new input
import sys
from typing import List, Tuple, Sequence, Optional, Dict
from typing_extensions import Literal
import pandas as pd
import numpy as np
import math
import time
import os
import matplotlib.pyplot as plt
import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import glob
import re
from sklearn import metrics
import argparse
from esm.modules import TransformerLayer, MultiheadAttention, ESM1bLayerNorm, ESM1LayerNorm, gelu
from multiprocessing import Pool
import seaborn as sns
import scipy
import pickle

sys.path.append('/share/terra/Users/gz2294/esm/')
from my_utils.dataset import DMSOneHotReprDataSet, SecStrucReprDataSet
from my_utils.optimizer import get_linear_schedule_with_warmup
from my_utils.TransformerLayers import GlobalCustomAttnTransformerLayer4, GlobalCustomAttnTransformerLayer5
from my_utils.plot import plot_aucs, plot_loss

WINDOW_SIZE = 500 * 2 + 1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class GELU(nn.Module):
    """
    Apply ESM's GELU activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SecStrucHead(nn.Module):
    """
    Secondary Structure classification problem is two heads
    Head 0 for secondary structure prediction
    Head 1 for RSA prediction
    """

    def __init__(self, task_in_dim: int, dropout: float):
        super(SecStrucHead, self).__init__()
        self.head = nn.ModuleList(
            (nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(task_in_dim, 8),
                nn.LogSoftmax(dim=-1)
            ), nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(task_in_dim, 1),
                nn.Sigmoid()
            )
            )
        )

    def forward(self, x):
        return self.head[0](x), self.head[1](x)


class VrtReprAgentBatchConverter(object):
    """
    Converts a batch of sequences to a batch of tokens
    """

    def __init__(self, alphabet, max_len=1001):
        self.alphabet = alphabet
        self.max_len = max_len

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        max_len = max(self.max_len, max(len(seq_encoded) for seq_encoded in seq_encoded_list))
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
                zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
            i,
            int(self.alphabet.prepend_bos): len(seq_encoded)
                                            + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens


class GraphAttnVrtOneHotReprAgent(nn.Module):
    """
    Fine tune ESM model to multi-task learning.
    """

    def __init__(self, language_model, language_model_repr_layer,
                 hidden_layer: Literal['attn', 'res', 'pass'],
                 tasks: List[Literal['secondary_struct', 'DMS', 'GoF_LoF', 'ClinVar']],
                 repr_actions: List[Literal['hidden_whole', 'hidden_pos_diff', 'hidden_pos_concat',
                                            'attn', 'attn_concat', 'attn_mask']],
                 task_out_dim=None,
                 repr_layer_type: Literal['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5'] = 'Layer5',
                 use_rotary_embeddings=True,
                 dropout: float = 0.1,
                 freeze_language_model=True):
        super(GraphAttnVrtOneHotReprAgent, self).__init__()
        self.language_model = language_model
        self.language_model_repr_layer = language_model_repr_layer
        embedding_dim = self.language_model.layers[self.language_model_repr_layer - 1].embed_dim
        # First check tasks
        assert len(tasks) == len(repr_actions), \
            "tasks, repr_actions must have the same length"
        if task_out_dim is None:
            task_out_dim = []
            for task in tasks:
                if task == 'secondary_struct':
                    task_out_dim.append(None)  # secondary structure does not require this
                elif task == 'DMS':
                    raise ValueError("DMS task requires task_out_dim")
                elif task == 'GoF_LoF':
                    task_out_dim.append(2)
                elif task == 'ClinVar':
                    task_out_dim.append(1)

        self.tasks = tasks
        self.repr_actions = repr_actions
        # Define hidden layer
        if hidden_layer == "attn":
            # ffn_embed_dim seems a magic number, find out if it is necessary
            # Turns out it somehow works as secondary structure prediction goes very well
            self.hidden_layer = TransformerLayer(embed_dim=embedding_dim,
                                                 ffn_embed_dim=5120, attention_heads=20, add_bias_kv=False,
                                                 use_rotary_embeddings=False)
        elif hidden_layer == "res":
            # TODO: implement res layer
            raise NotImplementedError("Residual layer is not implemented")
        elif hidden_layer == "pass":
            self.hidden_layer = nn.Identity()
        # Define task heads
        self.task_heads = nn.ModuleList()
        task_in_dim = []
        for repr_action in repr_actions:
            if repr_action.endswith("concat"):
                if repr_action == "attn_concat":
                    assert repr_layer_type == "Layer3", "attn_concat only works with Layer3"
                task_in_dim.append(embedding_dim * 2)
            else:
                task_in_dim.append(embedding_dim)
        for i, task in enumerate(tasks):
            if task == "secondary_struct":
                assert repr_actions[i].endswith("whole"), "Only whole representation is supported for secondary_struct"
                self.task_heads.append(
                    SecStrucHead(task_in_dim[i], dropout=dropout)
                )
            elif task == "DMS":
                self.task_heads.append(
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(task_in_dim[i], task_out_dim[i])
                    )
                )
            elif task == "GoF_LoF":
                self.task_heads.append(
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(task_in_dim[i], task_out_dim[i]),
                        nn.LogSoftmax(dim=-1)
                    )
                )
            elif task == "ClinVar":
                self.task_heads.append(
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(task_in_dim[i], task_out_dim[i])
                    )
                )
            else:
                raise ValueError(f"Invalid task {task}")
        # Then define attn layer, if required by repr_actions
        # TODO: implement attn layer
        # The original d_model in gMVP was 512, num_heads was 4
        # I am using a large fc layer while 20 heads like esm model
        if repr_layer_type == "Layer1":
            raise NotImplementedError("Layer1 is not Supported")
        elif repr_layer_type == "Layer2":
            raise NotImplementedError("Layer2 is not Supported")
        elif repr_layer_type == "Layer3":
            raise NotImplementedError("Layer3 is not Supported")
        elif repr_layer_type == "Layer4":
            self.repr_attn = GlobalCustomAttnTransformerLayer4(d_model=1280, d_pw=442, num_heads=20,
                                                               use_rotary_embeddings=use_rotary_embeddings)
        elif repr_layer_type == "Layer5":
            self.repr_attn = GlobalCustomAttnTransformerLayer5(d_model=1280, d_pw=442, num_heads=20,
                                                               use_rotary_embeddings=use_rotary_embeddings)
        else:
            raise ValueError(f"Invalid repr_layer_type {repr_layer_type}")
        # self.repr_attn = None
        # Freeze the language model if necessary
        self.freeze_language_model = freeze_language_model
        if freeze_language_model:
            for tag, value in self.language_model.named_parameters():
                # freeze all the parameters in the language model
                value.requires_grad = False

    def forward(self, batch_tokens, wt_aa=None, vr_aa=None, pos=None, MSA_features=None, MSA_masks=None,
                task_id=None):
        """
        Feed input to repr model and the classifier to compute logits.
        @param    batch_tokens_wt: an input array
        @param    batch_tokens_vr: an input array
        @param    pos: an input array, indicate the position of the token
        @param    task_id: the index of the task to be computed
        @param    MSA_features: an input array, indicate the features of the MSA
        @param    MSA_masks: an input array, indicate the masks of the MSA
        """
        # Feed input to esm
        if task_id is None:
            task_id = 0
        elif isinstance(task_id, torch.Tensor):
            task_id = int(task_id.cpu()[0])
        hidden = self.get_hidden(batch_tokens)
        if self.repr_actions[task_id].startswith("hidden"):
            if self.repr_actions[task_id] == "hidden_whole":
                # the hidden starts with <bos> and ends with <eos> so we need to remove them
                logits = self.task_heads[task_id](hidden[:, 1: batch_tokens.shape[1] - 1])
                return logits
            elif self.repr_actions[task_id] == "hidden_pos_diff":
                raise NotImplementedError("hidden_pos_diff is not implemented")
            elif self.repr_actions[task_id] == "hidden_pos_concat":
                raise NotImplementedError("hidden_pos_concat is not implemented")
        elif self.repr_actions[task_id].startswith("attn"):
            # TODO implement attn layer
            wt_aa_token = self.language_model.embed_tokens(wt_aa)
            vr_aa_token = self.language_model.embed_tokens(vr_aa)
            variant_hidden = self.get_variant_hidden(hidden[:, 1: batch_tokens.shape[1] - 1],
                                                     wt_aa_token, vr_aa_token,
                                                     pos, MSA_features, MSA_masks)
            variant_logits = self.task_heads[task_id](variant_hidden)
            return variant_logits

    def get_embedding(self, batch_tokens, wt_aa=None, vr_aa=None, pos=None,
                      MSA_features=None, MSA_masks=None, task_id=None):
        hidden = self.get_hidden(batch_tokens)
        if self.repr_actions[task_id].startswith("hidden"):
            if self.repr_actions[task_id] == "hidden_whole":
                raise NotImplementedError("hidden_whole is not implemented")
            elif self.repr_actions[task_id] == "hidden_pos_diff":
                raise NotImplementedError("hidden_pos_diff is not implemented")
            elif self.repr_actions[task_id] == "hidden_pos_concat":
                raise NotImplementedError("hidden_pos_concat is not implemented")
        elif self.repr_actions[task_id].startswith("attn"):
            # TODO implement attn layer
            wt_aa_token = self.language_model.embed_tokens(wt_aa)
            vr_aa_token = self.language_model.embed_tokens(vr_aa)
            variant_hidden = self.get_variant_hidden(hidden[:, 1: batch_tokens.shape[1] - 1],
                                                     wt_aa_token, vr_aa_token,
                                                     pos, MSA_features, MSA_masks)
        return variant_hidden

    def get_hidden(self, batch_tokens):
        """
        Get representations of the batch_tokens:
        """
        with torch.no_grad():
            repr = self.language_model(batch_tokens,
                                       repr_layers=[self.language_model_repr_layer],
                                       return_contacts=False)
        # get B x L X H representation
        repr = repr['representations'][self.language_model_repr_layer]
        # TODO: implement attn and concat
        # attn = repr['attentions']
        # concat = repr['contacts']
        if isinstance(self.hidden_layer, TransformerLayer):
            # hidden_layer is a transformer layer
            repr, attn = self.hidden_layer(repr)
        else:
            repr = self.hidden_layer(repr)
        # Ignore the attn and concat for now
        return repr

    def get_variant_hidden(self, hidden, wt_aa, vr_aa, pos, MSA_features, MSA_masks=None):
        var_repr = self.repr_attn(hidden, wt_aa, vr_aa, pos, MSA_features, MSA_masks)
        return var_repr

    def add_classifier(self, d_out, repr_action, activation):
        """
        Add a new classifier to the model.
        @param    d_out: the dimension of the output
        @param    activation: the activation function
        @param    repr_action: the repr_action
        """
        raise NotImplementedError("Not implemented yet")


class VrtReprAgentDataSet(Dataset):
    """
    Dataset for the VRT representation agent.
    """

    def __init__(self, data_path, batch_converter, batch_size, batch_number=None, shuffle=True, num_workers=32):
        """
        @param    data_path: the directory containing the data
        @param    batch_size: the batch size
        @param    batch_number: optional, the number of batches to load
        @param    shuffle: whether to shuffle the data
        @param    num_workers: the number of workers to use for data loading
        """
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        elif isinstance(data_path, str):
            self.data = pd.read_csv(data_path, index_col=0)
        else:
            raise ValueError("data_path must be a string or a pandas.DataFrame")
        if shuffle:
            self.data = self.data.sample(frac=1, random_state=0)
        if batch_size is None:
            assert batch_number is not None, "batch_size and batch_number cannot both be None"
            self.batch_size = int(np.ceil(self.data.shape[0] / batch_number))
            self.batch_number = batch_number
        else:
            self.batch_size = batch_size
            self.batch_number = int(np.ceil(self.data.shape[0] / self.batch_size))
        self.batch_converter = batch_converter
        self.num_workers = num_workers
        data_wt = tuple(zip(self.data['VarID'].astype('str'), self.data['wt'].astype('str')))
        # data_vr = tuple(zip(self.data['VarID'].astype('str'), self.data['sequence'].astype('str')))
        _, _, batch_tokens_wt = self.batch_converter(data_wt)
        # _, _, batch_tokens_vr = self.batch_converter(data_vr)
        ref_aa = tuple(zip(self.data['VarID'].astype('str'), self.data['ref'].astype('str')))
        alt_aa = tuple(zip(self.data['VarID'].astype('str'), self.data['alt'].astype('str')))
        _, _, batch_tokens_ref = self.batch_converter(ref_aa)
        _, _, batch_tokens_alt = self.batch_converter(alt_aa)

        labels = torch.tensor(self.data['score'].to_numpy(), dtype=torch.long)
        # note we use pos-1 as the variant position is 1-indexed
        pos = torch.tensor(self.data['pos'].to_numpy() - 1, dtype=torch.long)
        self.batch_tokens_wt = batch_tokens_wt
        self.wt_aa = batch_tokens_ref[:, [1]]
        self.vr_aa = batch_tokens_alt[:, [1]]
        self.labels = labels
        self.pos = pos
        # process MSA sequences
        with Pool(46) as pool:
            res = pool.starmap(self.parse_one_seq, zip(self.data['ENST'],
                                                       self.data['wt.orig'],
                                                       self.data['seq.start'],
                                                       self.data['seq.end']))
            mask = pool.starmap(self.parse_mask, zip(self.data['sequence.len']))
        self.msa_features = torch.tensor(np.array(res), dtype=torch.float)
        self.msa_masks = torch.tensor(np.array(mask), dtype=torch.float)

    @staticmethod
    def parse_one_seq(transcript, wt_orig, seq_start, seq_end, check_error=False):
        msa_alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWYU'))
        if pd.isna(transcript) or \
                not os.path.exists(f'data/MSA/{transcript}.pickle'):
            matched_line = False
            seq = ""
        else:
            with open(os.path.join('data/MSA/',
                                   transcript + '.pickle'), 'rb') as f:
                msa_mat = pickle.load(f)
            seq = ''.join(msa_alphabet[msa_mat[:, 0].astype(int)])
            matched_line = seq == wt_orig
        if matched_line:
            # crop the sequence to length of the data frame, and fill to 1001
            # R file parse seq_start starting from 1, so we need to minus 1
            msa_mat = msa_mat[seq_start - 1:seq_end, :221]
            msa_mat_seq = np.pad(msa_mat[:, [0]], ((0, 1001 - msa_mat.shape[0]), (0, 0)),
                                 'constant', constant_values=20)
            msa_mat_prob = np.pad(msa_mat[:, 1:21], ((0, 1001 - msa_mat.shape[0]), (0, 0)),
                                  'constant', constant_values=0)
            msa_mat_msa = np.pad(msa_mat[:, 21:221], ((0, 1001 - msa_mat.shape[0]), (0, 0)),
                                 'constant', constant_values=20)
            msa_mat = np.concatenate((msa_mat_seq, msa_mat_prob, msa_mat_msa), axis=1)
        else:
            msa_mat = np.array([[msa_alphabet.tolist().index(i) for i in wt_orig]])
            msa_mat = msa_mat[seq_start - 1:seq_end, [0]]
            msa_mat_seq = np.pad(msa_mat[:, [0]], ((0, 1001 - msa_mat.shape[0]), (0, 0)),
                                 'constant', constant_values=20)
            msa_mat_other = np.zeros((1001, 220))
            msa_mat = np.concatenate((msa_mat_seq, msa_mat_other), axis=1)
        if check_error:
            return msa_mat, matched_line, seq
        else:
            return msa_mat

    @staticmethod
    def parse_mask(seq_len):
        mask = np.ones((1001,), dtype=np.float32)
        seq_start = 0
        seq_end = seq_len
        # R file parse seq_start starting from 1, so we need to minus 1
        mask[seq_start:seq_end] = 0.0
        return mask

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.batch_tokens_wt[idx], self.wt_aa[idx], self.vr_aa[idx], \
               self.pos[idx], self.msa_features[idx], self.msa_masks[idx], \
               self.labels[idx]

    def count_labels(self):
        return self.data['score'].value_counts().sort_index().values

    def get_max_index(self):
        return int(np.ceil(len(self) / self.batch_size))

    def __iter__(self):
        return VrtReprAgentDataIterator(self)


class VrtReprAgentDataIterator:
    """
    Iterable class that returns batches of data.
    """

    def __init__(self, data_set: VrtReprAgentDataSet):
        """
        @param    data_set: the data set to iterate over
        """
        self._dataset = data_set
        self._index = 0
        self._max_index = int(np.ceil(len(self._dataset) / self._dataset.batch_size))
        self._batch_size = self._dataset.batch_size

    def __next__(self):
        """
        Returns the next batch of data.
        """
        if self._index < self._max_index:
            batch_idx = np.arange(self._index * self._batch_size,
                                  min((self._index + 1) * self._batch_size, len(self._dataset)))
            self._index += 1
            return self._dataset.batch_tokens_wt[batch_idx], \
                   self._dataset.wt_aa[batch_idx], self._dataset.vr_aa[batch_idx], \
                   self._dataset.pos[batch_idx], self._dataset.msa_features[batch_idx], \
                   self._dataset.msa_masks[batch_idx], self._dataset.labels[batch_idx]
        # End of Iteration
        raise StopIteration


class GraphAttnVrtReprAgentTrainer(object):
    """
    Train a VrtReprAgent model.
    """

    def __init__(self, language_model_name, language_model_repr_layer,
                 hidden_layer: Literal['attn', 'res', 'pass'],
                 tasks: List[Literal['secondary_struct', 'DMS', 'GoF_LoF', 'ClinVar']],
                 repr_actions: List[
                     Literal['hidden_whole', 'hidden_pos_diff', 'hidden_pos_concat', 'attn', 'attn_concat']],
                 train_data_files, test_data_files,
                 batch_sizes, batch_numbers=None,
                 task_out_dim=None,
                 build_datasets=True,
                 dropout: float = 0.1,
                 freeze_language_model=True,
                 lr=1e-5, min_lr_ratio=0.5,
                 save_dir=None, save_epochs=1, save_counters=None,
                 num_warmup_epochs=10, num_training_epochs=30,
                 device_id=None, data_distributed_parallel=True, seed=0):
        self.batch_converter = None
        if device_id is None:
            self.device = torch.device('cpu')
            self.device_id = device_id
            self.multi_gpu = False
            self.data_distributed_parallel = False
        elif isinstance(device_id, int):
            torch.cuda.set_per_process_memory_fraction(1.0, device_id)
            self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            self.device_id = device_id
            self.multi_gpu = False
            self.data_distributed_parallel = False
        else:
            assert torch.cuda.is_available(), "No GPU available"
            if not data_distributed_parallel:
                for device in device_id:
                    torch.cuda.set_per_process_memory_fraction(1.0, device)
            # use first device to store data and model
            self.device = f"cuda:{device_id[0]}"
            self.device_id = device_id
            self.multi_gpu = True
            self.data_distributed_parallel = data_distributed_parallel
        torch.cuda.empty_cache()
        self.seed = seed
        torch.manual_seed(seed)
        self.task_number = len(tasks)
        if not isinstance(batch_sizes, list):
            batch_sizes = [batch_sizes] * self.task_number
        if not isinstance(batch_numbers, list):
            batch_numbers = [batch_numbers] * self.task_number
        for batch_size, batch_number in zip(batch_sizes, batch_numbers):
            if batch_size is None:
                assert batch_number is not None, "batch_size and batch_number cannot both be None"
        self.train_data_sets = []
        self.test_data_sets = []
        self.batch_sizes = batch_sizes
        self.batch_numbers = batch_numbers
        self.loss_funcs = []
        assert len(train_data_files) == len(test_data_files) == len(tasks), \
            "Number of training, testing data files and tasks must be equal."
        self.tasks = tasks
        self.lr = lr
        self.num_warmup_epochs = num_warmup_epochs
        self.num_training_epochs = num_training_epochs
        self.min_lr_ratio = min_lr_ratio
        self.save_dir = save_dir
        self.save_epochs = save_epochs
        self.save_counters = save_counters
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "Log/"))
        self.writer_counter = 0
        self.epoch_counter = 0

        # first build model
        self.build_model(language_model_name, language_model_repr_layer,
                         hidden_layer, tasks, task_out_dim,
                         repr_actions, dropout, freeze_language_model)
        # then build data loaders
        if build_datasets:
            self.build_datasets(train_data_files, test_data_files)
        # then build loss functions
        for i in range(self.task_number):
            if self.tasks[i] == 'secondary_struct':
                self.loss_funcs.append([nn.NLLLoss(), nn.MSELoss()])
            elif self.tasks[i] == 'GoF_LoF':
                # to account for the imbalance of the GoF and LoF labels
                if self.train_data_sets[i] is not None:
                    weights = self.train_data_sets[i].count_labels()
                    weights = torch.tensor(np.max(weights) / weights, dtype=torch.float).to(self.device)
                    self.loss_funcs.append(nn.NLLLoss(weight=weights))
                else:
                    self.loss_funcs.append(nn.NLLLoss())
            elif self.tasks[i] == 'DMS':
                self.loss_funcs.append(nn.MSELoss())
            elif self.tasks[i] == 'ClinVar':
                self.loss_funcs.append(nn.BCEWithLogitsLoss())
        # then build optimizer and scheduler
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=self.lr,  # Default learning rate
                               eps=1e-8  # Default epsilon value
                               )
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=num_warmup_epochs,
                                                         num_training_steps=num_training_epochs,
                                                         minlr=self.min_lr_ratio)
        self.model.to(self.device)
        if device_id is not None and not isinstance(device_id, int) and not self.data_distributed_parallel:
            # if you use DDP, model should be defined here
            self.model = DataParallel(self.model, device_ids=device_id, output_device=self.device)

    def build_model(self,
                    language_model_name='esm1v',
                    language_model_repr_layer=33,
                    hidden_layer='attn',
                    tasks=None,
                    task_out_dim=None,
                    repr_actions=None,
                    dropout=0.1,
                    freeze_language_model=True):
        """
        Build the model.
        """
        if language_model_name == 'esm1v':
            model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5()
            self.model = GraphAttnVrtOneHotReprAgent(model, language_model_repr_layer,
                                                     hidden_layer=hidden_layer,
                                                     tasks=tasks,
                                                     task_out_dim=task_out_dim,
                                                     repr_actions=repr_actions,
                                                     dropout=dropout,
                                                     freeze_language_model=freeze_language_model)
            self.batch_converter = VrtReprAgentBatchConverter(alphabet)
        elif language_model_name == 'esm1b':
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            self.model = GraphAttnVrtOneHotReprAgent(model, language_model_repr_layer,
                                                     hidden_layer=hidden_layer,
                                                     tasks=tasks,
                                                     task_out_dim=task_out_dim,
                                                     repr_actions=repr_actions,
                                                     dropout=dropout,
                                                     freeze_language_model=freeze_language_model)
            self.batch_converter = VrtReprAgentBatchConverter(alphabet)
        elif language_model_name == 'esm2':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = GraphAttnVrtOneHotReprAgent(model, language_model_repr_layer,
                                                     hidden_layer=hidden_layer,
                                                     tasks=tasks,
                                                     task_out_dim=task_out_dim,
                                                     repr_actions=repr_actions,
                                                     dropout=dropout,
                                                     freeze_language_model=freeze_language_model)
            self.batch_converter = VrtReprAgentBatchConverter(alphabet)
        elif language_model_name == 'esm1b.secstruc.CHPs':
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            self.model = GraphAttnVrtOneHotReprAgent(model, language_model_repr_layer,
                                                     hidden_layer=hidden_layer,
                                                     tasks=tasks,
                                                     task_out_dim=task_out_dim,
                                                     repr_actions=repr_actions,
                                                     dropout=dropout,
                                                     freeze_language_model=freeze_language_model)
            self.model.to(self.device)
            self.batch_converter = VrtReprAgentBatchConverter(alphabet)
            self.model.hidden_layer.load_state_dict(torch.load(
                os.path.join("model_checkpoints/"
                             "gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.CHPs/model.best.hidden_layer.pt"),
                map_location=self.device)
            )
            self.model.repr_attn.load_state_dict(
                torch.load(
                    os.path.join("model_checkpoints/"
                                 "gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.CHPs/model.best.repr_attn.pt"),
                    map_location=self.device
                )
            )
            if "secondary_struct" in tasks:
                self.model.task_heads[np.where(np.array(tasks) == "secondary_struct")[0][0]].load_state_dict(
                    torch.load(
                        os.path.join("model_checkpoints/"
                                     "gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.CHPs/model.best.task_heads.secstruc.pt"),
                        map_location=self.device
                    )
                )
            if "ClinVar" in tasks:
                self.model.task_heads[np.where(np.array(tasks) == "ClinVar")[0][0]].load_state_dict(
                    torch.load(
                        os.path.join("model_checkpoints/"
                                     "gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.CHPs/model.best.task_heads.CHPs.pt"),
                        map_location=self.device
                    )
                )
        elif language_model_name == 'esm1b.secstruc':
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            self.model = GraphAttnVrtOneHotReprAgent(model, language_model_repr_layer,
                                                     hidden_layer=hidden_layer,
                                                     tasks=tasks,
                                                     task_out_dim=task_out_dim,
                                                     repr_actions=repr_actions,
                                                     dropout=dropout,
                                                     freeze_language_model=freeze_language_model)
            self.model.to(self.device)
            self.batch_converter = VrtReprAgentBatchConverter(alphabet)
            self.model.hidden_layer.load_state_dict(torch.load(
                os.path.join("model_checkpoints/"
                             "gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.CHPs/model.best.hidden_layer.pt"),
                map_location=self.device)
            )
        else:
            raise NotImplementedError

    def build_datasets(self, train_data_files, test_data_files):
        for i, (train_file, test_file) in enumerate(zip(train_data_files, test_data_files)):
            if self.tasks[i] == 'secondary_struct':
                self.train_data_sets.append(
                    SecStrucReprDataSet(train_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                        batch_converter=self.batch_converter)
                    if train_file is not None else None
                )
                self.test_data_sets.append(
                    SecStrucReprDataSet(test_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                        shuffle=False,
                                        batch_converter=self.batch_converter)
                    if test_file is not None else None
                )
            elif self.tasks[i] == 'DMS':
                self.train_data_sets.append(
                    DMSOneHotReprDataSet(train_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                         batch_converter=self.batch_converter)
                    if train_file is not None else None
                )
                self.test_data_sets.append(
                    DMSOneHotReprDataSet(test_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                         shuffle=False,
                                         batch_converter=self.batch_converter)
                    if test_file is not None else None
                )
            else:
                self.train_data_sets.append(
                    VrtReprAgentDataSet(train_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                        batch_converter=self.batch_converter)
                    if train_file is not None else None
                )
                self.test_data_sets.append(
                    VrtReprAgentDataSet(test_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                        shuffle=False,
                                        batch_converter=self.batch_converter)
                    if test_file is not None else None
                )

    def build_loss_funcs(self):
        for i in range(self.task_number):
            if self.tasks[i] == 'secondary_struct':
                self.loss_funcs.append([nn.NLLLoss(), nn.MSELoss()])
            elif self.tasks[i] == 'GoF_LoF':
                # to account for the imbalance of the GoF and LoF labels
                weights = self.train_data_sets[i].count_labels()
                weights = torch.tensor(np.max(weights) / weights, dtype=torch.float).to(self.device)
                self.loss_funcs.append(nn.NLLLoss(weight=weights))
            elif self.tasks[i] == 'DMS':
                self.loss_funcs.append(nn.MSELoss())
            elif self.tasks[i] == 'ClinVar':
                # to account for the imbalance of the labels, BCEWithLogitsLoss recommended ratio of neg / pos
                weights = self.train_data_sets[i].count_labels()
                weights = torch.tensor(weights[0] / weights[1], dtype=torch.float).to(self.device)
                self.loss_funcs.append(nn.BCEWithLogitsLoss(pos_weight=weights))

    def build_train_datasets(self, train_data_files):
        for i, train_file in enumerate(train_data_files):
            if self.tasks[i] == 'secondary_struct':
                self.train_data_sets.append(
                    SecStrucReprDataSet(train_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                        batch_converter=self.batch_converter))
            elif self.tasks[i] == 'DMS':
                self.train_data_sets.append(
                    DMSOneHotReprDataSet(train_file, batch_size=self.batch_sizes[i], batch_number=self.batch_numbers[i],
                                         batch_converter=self.batch_converter))
            else:
                self.train_data_sets.append(
                    VrtReprAgentDataSet(train_file, batch_size=self.batch_sizes[i],
                                        batch_number=self.batch_numbers[i],
                                        batch_converter=self.batch_converter)
                )

    def build_test_datasets(self, test_data_files, batch_size, batch_number):
        for i, test_file in enumerate(test_data_files):
            if self.tasks[i] == 'secondary_struct':
                self.test_data_sets.append(
                    SecStrucReprDataSet(test_file, batch_size=batch_size, batch_number=batch_number,
                                        batch_converter=self.batch_converter))
            else:
                self.test_data_sets.append(
                    VrtReprAgentDataSet(test_file, batch_size=batch_size,
                                        batch_number=batch_number,
                                        shuffle=False,
                                        batch_converter=self.batch_converter)
                )

    def single_gpu_one_epoch(self, task_ids=None):
        # zero the gradients before each epoch
        self.model.zero_grad()
        self.model.train()
        # for warm up epochs, do not update the repr model but just classifier
        if task_ids is None:
            task_ids = np.arange(self.task_number).tolist()
            tasks = self.task_number
        else:
            tasks = len(task_ids)
        losses = [[] for k in range(tasks)]
        # TODO: Decide let all tasks run same batches or not
        # Trick here is, for one batch, we wait until all tasks has finished to calculate the gradient,
        # then we update the weights.
        task_finished = [False for k in range(tasks)]
        batch_count = 0
        iter_data_loader = [iter(self.train_data_sets[i]) for i in task_ids]
        while sum(task_finished) < tasks:
            batch_start_time = time.time()
            loss_sum = None
            for i, task_id in enumerate(task_ids):
                if task_finished[i]:
                    continue
                if self.tasks[task_id] == 'secondary_struct':
                    try:
                        batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    batch_tokens = batch_tokens.to(self.device)
                    labels_ss = labels_ss.to(self.device)
                    labels_rsa = labels_rsa.to(self.device)
                    logits = self.model(batch_tokens, task_id=task_id)
                    loss_ss = self.loss_funcs[task_id][0](
                        logits[0].reshape(-1, logits[0].shape[-1]), labels_ss.reshape(-1)
                    )
                    loss_rsa = self.loss_funcs[task_id][1](logits[1][:, :, 0], labels_rsa)
                    self.write_loss([loss_ss.item(), loss_rsa.item()], task_id)
                    loss = loss_ss + loss_rsa
                else:
                    try:
                        batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, labels = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    batch_tokens_wt = batch_tokens_wt.to(self.device)
                    wt_aa = wt_aa.to(self.device)
                    vr_aa = vr_aa.to(self.device)
                    pos = pos.to(self.device)
                    msa_features = msa_features.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, task_id)
                    if self.tasks[task_id] == 'ClinVar':
                        logits = logits[:, 0]
                        labels = labels.to(torch.float)
                    loss = self.loss_funcs[task_id](logits, labels)
                    self.write_loss(loss.item(), task_id)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
                losses[i].append(loss.item())

            # step optimizer after all tasks have finished
            if loss_sum is None:
                break
            loss_sum.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer_counter += 1
            batch_end_time = time.time()
            if self.save_counters is not None and self.writer_counter % self.save_counters == 0:
                self.save_model(whole_model=False, counter_id=self.writer_counter)
            print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
            batch_count += 1
        # step scheduler after all batches have finished
        self.scheduler.step()
        self.epoch_counter += 1
        for i, task_id in enumerate(task_ids):
            print(f"Task {task_id} finished {len(losses[i])} batches with ",
                  f"loss: {np.mean(losses[i]):.4f}")
        if self.epoch_counter % self.save_epochs == 0:
            self.save_model(whole_model=True)
        losses = [np.mean(losses[i]) for i in range(tasks)]
        return losses

    def data_parallel_gpu_one_epoch(self, task_ids=None):
        # zero the gradients before each epoch
        self.model.zero_grad()
        self.model.train()
        # for warm up epochs, do not update the repr model but just classifier
        if task_ids is None:
            task_ids = np.arange(self.task_number).tolist()
            tasks = self.task_number
        else:
            tasks = len(task_ids)
        losses = [[] for k in range(tasks)]
        # TODO: Decide let all tasks run same batches or not
        # Trick here is, for one batch, we wait until all tasks has finished to calculate the gradient,
        # then we update the weights.
        task_finished = [False for k in range(tasks)]
        batch_count = 0
        iter_data_loader = [iter(self.train_data_sets[i]) for i in task_ids]
        while sum(task_finished) < tasks:
            batch_start_time = time.time()
            loss_sum = None
            for i, task_id in enumerate(task_ids):
                if task_finished[i]:
                    continue
                if self.tasks[task_id] == 'secondary_struct':
                    try:
                        batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    labels_ss = labels_ss.to(self.device)
                    labels_rsa = labels_rsa.to(self.device)
                    task_id_tensor = torch.tensor([task_id] * batch_tokens.shape[0], dtype=torch.int)
                    logits = self.model(batch_tokens, task_id=task_id_tensor)
                    loss_ss = self.loss_funcs[task_id][0](
                        logits[0].reshape(-1, logits[0].shape[-1]), labels_ss.reshape(-1)
                    )
                    loss_rsa = self.loss_funcs[task_id][1](logits[1][:, :, 0], labels_rsa)
                    self.write_loss([loss_ss.item(), loss_rsa.item()], task_id)
                    loss = loss_ss + loss_rsa
                else:
                    try:
                        batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, labels = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    task_id_tensor = torch.tensor([task_id] * batch_tokens_wt.shape[0], dtype=torch.int)
                    labels = labels.to(self.device)
                    logits = self.model(batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, task_id_tensor)
                    if self.tasks[task_id] == 'ClinVar':
                        logits = logits[:, 0]
                        labels = labels.to(torch.float)
                    loss = self.loss_funcs[task_id](logits, labels)
                    self.write_loss(loss.item(), task_id)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
                losses[i].append(loss.item())

            # step optimizer after all tasks have finished
            if loss_sum is None:
                break
            loss_sum.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer_counter += 1
            batch_end_time = time.time()
            if self.save_counters is not None and self.writer_counter % self.save_counters == 0:
                self.save_model(whole_model=False, counter_id=self.writer_counter)
            print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
            batch_count += 1
        # step scheduler after all batches have finished
        self.scheduler.step()
        self.epoch_counter += 1
        for i, task_id in enumerate(task_ids):
            print(f"Task {task_id} finished {len(losses[i])} batches with ",
                  f"loss: {np.mean(losses[i]):.4f}")
        if self.epoch_counter % self.save_epochs == 0:
            self.save_model(whole_model=True)
        losses = [np.mean(losses[i]) for i in range(tasks)]
        return losses

    def data_distributed_parallel_gpu_one_epoch(self, rank, world_size, task_ids):
        # zero the gradients before each epoch
        self.model.zero_grad()
        self.model.train()
        # set up training processes
        setup(rank, world_size)
        device = f'cuda:{rank}'
        # if task_ids is None, then train on all tasks
        if task_ids is None:
            task_ids = np.arange(self.task_number).tolist()
            tasks = self.task_number
        else:
            tasks = len(task_ids)
        # set up model to the correct device and set up ddp model
        model = self.model.to(device)
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)
        losses = [[] for k in range(tasks)]
        # TODO: Decide let all tasks run same batches or not
        # Trick here is, for one batch, we wait until all tasks has finished to calculate the gradient,
        # then we update the weights.
        task_finished = [False for k in range(tasks)]
        batch_count = 0
        # set up data sampler
        sampler = [DistributedSampler(self.train_data_sets[i],
                                      num_replicas=world_size,
                                      rank=rank,
                                      shuffle=True,  # May be True
                                      seed=0) for i in task_ids]
        iter_data_loader = [iter(DataLoader(self.train_data_sets[i],
                                            batch_size=self.train_data_sets[i].batch_size,
                                            shuffle=False,  # Must be False!
                                            num_workers=10,
                                            sampler=sampler[j],
                                            pin_memory=True)) for j, i in enumerate(task_ids)]
        while sum(task_finished) < tasks:
            batch_start_time = time.time()
            loss_sum = None
            for i, task_id in enumerate(task_ids):
                if task_finished[i]:
                    continue
                if self.tasks[task_id] == 'secondary_struct':
                    try:
                        batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    batch_tokens = batch_tokens.to(device)
                    labels_ss = labels_ss.to(device)
                    labels_rsa = labels_rsa.to(device)
                    logits = ddp_model(batch_tokens, task_id=task_id)
                    loss_ss = self.loss_funcs[task_id][0](
                        logits[0].reshape(-1, logits[0].shape[-1]), labels_ss.reshape(-1)
                    )
                    loss_rsa = self.loss_funcs[task_id][1](logits[1][:, :, 0], labels_rsa)
                    # self.write_loss([loss_ss.item(), loss_rsa.item()], task_id, rank_id=rank)
                    loss = loss_ss + loss_rsa
                else:
                    try:
                        batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, labels = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    batch_tokens_wt = batch_tokens_wt.to(device)
                    wt_aa = wt_aa.to(device)
                    vr_aa = vr_aa.to(device)
                    pos = pos.to(device)
                    msa_features = msa_features.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)
                    logits = self.model(batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, task_id)
                    if self.tasks[task_id] == 'ClinVar':
                        logits = logits[:, 0]
                        labels = labels.to(torch.float)
                    loss = self.loss_funcs[task_id](logits, labels)
                    # self.write_loss(loss.item(), task_id, rank_id=rank)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
                losses[i].append(loss.item())
            # step optimizer after all tasks have finished
            if loss_sum is None:
                break
            loss_sum.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if rank == 0:
                self.writer_counter += 1
            batch_end_time = time.time()
            if rank == 0 and self.save_counters is not None and self.writer_counter % self.save_counters == 0:
                self.save_model(whole_model=False, counter_id=self.writer_counter)
            # wait for all ranks to update weights before next batch
            dist.barrier()
            print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
            batch_count += 1
        # step scheduler after all batches have finished
        # only rank 0 updates the scheduler, epoch_counter
        if rank == 0:
            self.scheduler.step()
            self.epoch_counter += 1
        for i, task_id in enumerate(task_ids):
            print(f"Task {task_id} finished {len(losses[i])} batches with ",
                  f"loss: {np.mean(losses[i]):.4f}")
        if rank == 0 and self.epoch_counter % self.save_epochs == 0:
            self.save_model(whole_model=True)
        losses = [np.mean(losses[i]) for i in range(tasks)]
        cleanup()
        return losses

    def train_one_epoch(self, task_ids=None):
        if self.multi_gpu:
            if self.data_distributed_parallel:
                world_size = len(self.device_id)
                losses = mp.spawn(
                    self.data_distributed_parallel_gpu_one_epoch,
                    args=(world_size, task_ids),
                    nprocs=world_size,
                    join=True)
                self.load_model(whole_model=True, epoch_id=self.epoch_counter)
                return losses
            else:
                return self.data_parallel_gpu_one_epoch(task_ids)
        else:
            return self.single_gpu_one_epoch(task_ids)

    def fine_tune_one_epoch(self, task_ids=None, fine_tune_task_ids=None):
        # Only support one gpu for fine-tuning
        # for fine tune job, we want to keep all previous task losses.
        # But we don't use all of them but just use equal sample sizes as fine_tune_task_ids until they finish.
        self.model.zero_grad()
        torch.manual_seed(self.seed)
        # for fine tune model, hidden layer is fixed, only update the attn layer
        for _, param in self.model.hidden_layer.named_parameters():
            param.requires_grad = False
        self.model.train()
        # for warm up epochs, do not update the repr model but just classifier
        if task_ids is None:
            task_ids = np.arange(self.task_number).tolist()
            tasks = self.task_number
        else:
            tasks = len(task_ids)
        if fine_tune_task_ids is None:
            fine_tune_task_ids = np.arange(self.task_number).tolist()
        losses = [[] for k in range(tasks)]
        # Trick here is, for one batch, we wait until all tasks has finished to calculate the gradient,
        # then we update the weights.
        task_finished = [False for k in range(tasks)]
        fine_tune_task_finished = [id not in fine_tune_task_ids for id in task_ids]
        batch_count = 0
        iter_data_loader = [iter(DataLoader(self.train_data_sets[i],
                                            batch_size=self.train_data_sets[i].batch_size,
                                            shuffle=True,  # Must be TRUE so that each time doing shuffle!
                                            num_workers=20,
                                            pin_memory=True)) for j, i in enumerate(task_ids)]
        while sum(fine_tune_task_finished) < tasks:
            batch_start_time = time.time()
            loss_sum = None
            for i, task_id in enumerate(task_ids):
                if task_finished[i]:
                    continue
                if self.tasks[task_id] == 'secondary_struct':
                    try:
                        batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        fine_tune_task_finished[i] = True
                        continue
                    batch_tokens = batch_tokens.to(self.device)
                    labels_ss = labels_ss.to(self.device)
                    labels_rsa = labels_rsa.to(self.device)
                    logits = self.model(batch_tokens, task_id=task_id)
                    loss_ss = self.loss_funcs[task_id][0](
                        logits[0].reshape(-1, logits[0].shape[-1]), labels_ss.reshape(-1)
                    )
                    loss_rsa = self.loss_funcs[task_id][1](logits[1][:, :, 0], labels_rsa)
                    self.write_loss([loss_ss.item(), loss_rsa.item()], task_id)
                    loss = loss_ss + loss_rsa
                else:
                    try:
                        batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, labels = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        fine_tune_task_finished[i] = True
                        continue
                    batch_tokens_wt = batch_tokens_wt.to(self.device)
                    wt_aa = wt_aa.to(self.device)
                    vr_aa = vr_aa.to(self.device)
                    pos = pos.to(self.device)
                    msa_features = msa_features.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, task_id)
                    print(logits.detach().cpu().numpy().max())
                    if self.tasks[task_id] == 'ClinVar':
                        logits = logits[:, 0]
                        labels = labels.to(torch.float)
                    loss = self.loss_funcs[task_id](logits, labels)
                    self.write_loss(loss.item(), task_id)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
                losses[i].append(loss.item())
            # step optimizer after all tasks have finished
            if loss_sum is None:
                break
            loss_sum.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer_counter += 1
            batch_end_time = time.time()
            if self.save_counters is not None and self.writer_counter % self.save_counters == 0:
                self.save_model(whole_model=False, counter_id=self.writer_counter)
            print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
            batch_count += 1
        # step scheduler after all batches have finished
        self.scheduler.step()
        self.epoch_counter += 1
        for i, task_id in enumerate(task_ids):
            print(f"Task {task_id} finished {len(losses[i])} batches with ",
                  f"loss: {np.mean(losses[i]):.4f}")
        if self.epoch_counter % self.save_epochs == 0:
            self.save_model(whole_model=True)
        losses = [np.mean(losses[i]) for i in range(tasks)]
        return losses

    def fine_tune_no_pretrain_one_epoch(self, task_ids=None, fine_tune_task_ids=None):
        # Only support one gpu for fine-tuning
        # for fine tune job, we want to keep all previous task losses.
        # But we don't use all of them but just use equal sample sizes as fine_tune_task_ids until they finish.
        self.model.zero_grad()
        torch.manual_seed(self.seed)
        # for fine tune model, hidden layer is fixed, only update the attn layer
        # for _, param in self.model.hidden_layer.named_parameters():
        #     param.requires_grad = False
        self.model.train()
        # for warm up epochs, do not update the repr model but just classifier
        if task_ids is None:
            task_ids = np.arange(self.task_number).tolist()
            tasks = self.task_number
        else:
            tasks = len(task_ids)
        if fine_tune_task_ids is None:
            fine_tune_task_ids = np.arange(self.task_number).tolist()
        losses = [[] for k in range(tasks)]
        # Trick here is, for one batch, we wait until all tasks has finished to calculate the gradient,
        # then we update the weights.
        task_finished = [False for k in range(tasks)]
        fine_tune_task_finished = [id not in fine_tune_task_ids for id in task_ids]
        batch_count = 0
        iter_data_loader = [iter(DataLoader(self.train_data_sets[i],
                                            batch_size=self.train_data_sets[i].batch_size,
                                            shuffle=True,  # Must be TRUE so that each time doing shuffle!
                                            num_workers=20,
                                            pin_memory=True)) for j, i in enumerate(task_ids)]
        while sum(fine_tune_task_finished) < tasks:
            batch_start_time = time.time()
            loss_sum = None
            for i, task_id in enumerate(task_ids):
                if task_finished[i]:
                    continue
                if self.tasks[task_id] == 'secondary_struct':
                    try:
                        batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        fine_tune_task_finished[i] = True
                        continue
                    batch_tokens = batch_tokens.to(self.device)
                    labels_ss = labels_ss.to(self.device)
                    labels_rsa = labels_rsa.to(self.device)
                    logits = self.model(batch_tokens, task_id=task_id)
                    loss_ss = self.loss_funcs[task_id][0](
                        logits[0].reshape(-1, logits[0].shape[-1]), labels_ss.reshape(-1)
                    )
                    loss_rsa = self.loss_funcs[task_id][1](logits[1][:, :, 0], labels_rsa)
                    self.write_loss([loss_ss.item(), loss_rsa.item()], task_id)
                    loss = loss_ss + loss_rsa
                else:
                    try:
                        batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, labels = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        fine_tune_task_finished[i] = True
                        continue
                    batch_tokens_wt = batch_tokens_wt.to(self.device)
                    wt_aa = wt_aa.to(self.device)
                    vr_aa = vr_aa.to(self.device)
                    pos = pos.to(self.device)
                    msa_features = msa_features.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, task_id)
                    print(logits.detach().cpu().numpy().max())
                    if self.tasks[task_id] == 'ClinVar':
                        logits = logits[:, 0]
                        labels = labels.to(torch.float)
                    loss = self.loss_funcs[task_id](logits, labels)
                    self.write_loss(loss.item(), task_id)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
                losses[i].append(loss.item())
            # step optimizer after all tasks have finished
            if loss_sum is None:
                break
            loss_sum.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer_counter += 1
            batch_end_time = time.time()
            if self.save_counters is not None and self.writer_counter % self.save_counters == 0:
                self.save_model(whole_model=False, counter_id=self.writer_counter)
            print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
            batch_count += 1
        # step scheduler after all batches have finished
        self.scheduler.step()
        self.epoch_counter += 1
        for i, task_id in enumerate(task_ids):
            print(f"Task {task_id} finished {len(losses[i])} batches with ",
                  f"loss: {np.mean(losses[i]):.4f}")
        if self.epoch_counter % self.save_epochs == 0:
            self.save_model(whole_model=True)
        losses = [np.mean(losses[i]) for i in range(tasks)]
        return losses

    def test_one_epoch(self, task_ids=None):
        # zero the gradients before each epoch
        self.model.zero_grad()
        self.model.eval()
        if task_ids is None:
            task_ids = np.arange(self.task_number).tolist()
            tasks = self.task_number
        else:
            tasks = len(task_ids)
        results_dfs = [pd.DataFrame() for k in range(tasks)]
        labels_dfs = [pd.DataFrame() for k in range(tasks)]
        task_finished = [False for k in range(tasks)]
        batch_count = 0
        with torch.no_grad():
            iter_data_loader = [iter(self.test_data_sets[i]) for i in task_ids]
            while sum(task_finished) < tasks:
                batch_start_time = time.time()
                for i, task_id in enumerate(task_ids):
                    if self.tasks[task_id] == 'secondary_struct':
                        try:
                            batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                        except StopIteration:
                            task_finished[i] = True
                            continue
                        batch_tokens = batch_tokens.to(self.device)
                        labels_ss = labels_ss.to(self.device)
                        labels_rsa = labels_rsa.to(self.device)
                        labels = torch.cat((torch.nn.functional.one_hot(labels_ss.reshape(-1)),
                                            labels_rsa.reshape(-1, 1)), dim=-1)
                        logits_ss, logits_rsa = self.model(batch_tokens, task_id=task_id)
                        logits = torch.cat((torch.exp(logits_ss), logits_rsa), dim=-1).reshape(-1, 9)
                    else:
                        try:
                            batch_tokens_wt, wt_aa, alt_aa, pos, msa_features, masks, labels = next(
                                iter_data_loader[i])
                        except StopIteration:
                            task_finished[i] = True
                            continue
                        batch_tokens_wt = batch_tokens_wt.to(self.device)
                        wt_aa = wt_aa.to(self.device)
                        alt_aa = alt_aa.to(self.device)
                        pos = pos.to(self.device)
                        msa_features = msa_features.to(self.device)
                        masks = masks.to(self.device)
                        labels = labels.to(self.device)
                        logits = self.model(batch_tokens_wt, wt_aa, alt_aa, pos, msa_features, masks, task_id)
                        if self.tasks[task_id] == 'ClinVar':
                            logits = torch.sigmoid(logits[:, 0])
                        elif self.tasks[task_id] == 'GoF_LoF':
                            logits = torch.exp(logits)
                    labels = labels.detach().cpu().numpy()
                    logits = logits.detach().cpu().numpy()
                    results_dfs[i] = pd.concat([results_dfs[i],
                                                pd.DataFrame(logits)], axis=0)
                    labels_dfs[i] = pd.concat([labels_dfs[i],
                                               pd.DataFrame(labels)], axis=0)
                batch_end_time = time.time()
                print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
                batch_count += 1
        return results_dfs, labels_dfs

    def get_embedding_one_epoch(self, task_ids=None):
        # zero the gradients before each epoch
        self.model.zero_grad()
        self.model.eval()
        if task_ids is None:
            task_ids = np.arange(self.task_number).tolist()
            tasks = self.task_number
        else:
            tasks = len(task_ids)
        results_dfs = [pd.DataFrame() for k in range(tasks)]
        labels_dfs = [pd.DataFrame() for k in range(tasks)]
        task_finished = [False for k in range(tasks)]
        batch_count = 0
        with torch.no_grad():
            iter_data_loader = [iter(self.test_data_sets[i]) for i in task_ids]
            while sum(task_finished) < tasks:
                batch_start_time = time.time()
                for i, task_id in enumerate(task_ids):
                    if self.tasks[task_id] == 'secondary_struct':
                        try:
                            batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                        except StopIteration:
                            task_finished[i] = True
                            continue
                        batch_tokens = batch_tokens.to(self.device)
                        labels_ss = labels_ss.to(self.device)
                        labels_rsa = labels_rsa.to(self.device)
                        labels = torch.cat((torch.nn.functional.one_hot(labels_ss.reshape(-1)),
                                            labels_rsa.reshape(-1, 1)), dim=-1)
                        logits_ss, logits_rsa = self.model(batch_tokens, task_id=task_id)
                        logits = torch.cat((torch.exp(logits_ss), logits_rsa), dim=-1).reshape(-1, 9)
                    else:
                        try:
                            batch_tokens_wt, wt_aa, alt_aa, pos, msa_features, masks, labels = next(
                                iter_data_loader[i])
                        except StopIteration:
                            task_finished[i] = True
                            continue
                        batch_tokens_wt = batch_tokens_wt.to(self.device)
                        wt_aa = wt_aa.to(self.device)
                        alt_aa = alt_aa.to(self.device)
                        pos = pos.to(self.device)
                        msa_features = msa_features.to(self.device)
                        masks = masks.to(self.device)
                        labels = labels.to(self.device)
                        logits = self.model.get_embedding(batch_tokens_wt, wt_aa, alt_aa,
                                                          pos, msa_features, masks, task_id)
                    labels = labels.detach().cpu().numpy()
                    logits = logits.detach().cpu().numpy()
                    results_dfs[i] = pd.concat([results_dfs[i],
                                                pd.DataFrame(logits)], axis=0)
                    labels_dfs[i] = pd.concat([labels_dfs[i],
                                               pd.DataFrame(labels)], axis=0)
                batch_end_time = time.time()
                print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
                batch_count += 1
        return results_dfs, labels_dfs

    def write_loss(self, loss, task_id, rank_id=None):
        if self.data_distributed_parallel:
            assert rank_id is not None
            rank_id = f"_{rank_id}"
        else:
            rank_id = ""
        if self.tasks[task_id] == 'secondary_struct':
            self.writer.add_scalar(f"loss/task_{self.tasks[task_id]}_ss{rank_id}", loss[0], self.writer_counter)
            self.writer.add_scalar(f"loss/task_{self.tasks[task_id]}_rsa{rank_id}", loss[1], self.writer_counter)
        else:
            self.writer.add_scalar(f"loss/task_{self.tasks[task_id]}{rank_id}", loss, self.writer_counter)

    def save_model(self, whole_model=False, task_id=None, counter_id=None):
        if counter_id is not None:
            counter_id = str(counter_id) + "."
        else:
            counter_id = ""
        if not self.model.freeze_language_model:
            print("Representation model is not frozen, saving whole model anyway")
            whole_model = True
        if whole_model:
            torch.save(self.model.state_dict(),
                       os.path.join(self.save_dir, f"model.{self.epoch_counter}.{counter_id}pt"))
        else:
            # TODO: implement when len(repr_layers) ≥ 2
            torch.save(self.model.hidden_layer.state_dict(),
                       os.path.join(self.save_dir,
                                    f"model.{self.epoch_counter}.{counter_id}hidden_layer.pt"))
            torch.save(self.model.repr_attn.state_dict(),
                       os.path.join(self.save_dir,
                                    f"model.{self.epoch_counter}.{counter_id}repr_attn.pt"))
            if task_id is None:
                torch.save(self.model.task_heads.state_dict(),
                           os.path.join(self.save_dir, f"model.{self.epoch_counter}.{counter_id}task_heads.pt"))
            else:
                torch.save(self.model.task_heads[task_id].state_dict(),
                           os.path.join(self.save_dir,
                                        f"model.{self.epoch_counter}.{counter_id}task_heads.{task_id}.pt"))

    def load_model(self, whole_model=False, task_id=None, epoch_id=None, counter_id=None, step_scheduler=False):
        if epoch_id is None:
            pt_files = glob.glob(f"{self.save_dir}/model.*.pt", recursive=True)
            pr_number = [re.findall('[0-9]+', string) for string in pt_files]
            epoch_id = max(pr_number)
        if counter_id is not None:
            self.writer_counter = counter_id
            counter_id = str(counter_id) + "."
        else:
            if step_scheduler:
                non_empty_train_sets = [self.train_data_sets[i].get_max_index() for i in range(self.task_number)
                                        if self.train_data_sets[i] is not None]
                if len(non_empty_train_sets) > 0:
                    self.writer_counter = (epoch_id + 1) * max(non_empty_train_sets)
            counter_id = ""
        # set self.epoch_counter to the epoch_id
        self.epoch_counter = epoch_id
        if whole_model:
            self.model.load_state_dict(torch.load(os.path.join(self.save_dir, f"model.{epoch_id}.{counter_id}pt"),
                                                  map_location=self.device))
        else:
            # TODO: implement when len(repr_layers) ≥ 2
            self.model.hidden_layer.load_state_dict(torch.load(
                os.path.join(self.save_dir, f"model.{self.epoch_counter}.{counter_id}hidden_layer.pt"),
                map_location=self.device))
            self.model.repr_attn.load_state_dict(torch.load(
                os.path.join(self.save_dir, f"model.{self.epoch_counter}.{counter_id}repr_attn.pt"),
                map_location=self.device))
            if task_id is None:
                self.model.task_heads.load_state_dict(torch.load(
                    os.path.join(self.save_dir, f"model.{epoch_id}.{counter_id}task_heads.pt"),
                    map_location=self.device))
            else:
                self.model.task_heads[task_id].load_state_dict(torch.load(
                    os.path.join(self.save_dir, f"model.{epoch_id}.{counter_id}task_heads.{task_id}.pt"),
                    map_location=self.device))
        if step_scheduler:
            for i in range(self.epoch_counter):
                self.scheduler.step()


def train_secondary_structures(ids,
                               language_model_name='esm1v',
                               model_dir='gMVP.style.esm.secstruc/',
                               device_id=1,
                               batch_size=None,
                               batch_number=5,
                               warmup_epochs=10,
                               epochs=30):
    # create output directory
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir), exist_ok=True)
    # Load pretrained model
    if isinstance(device_id, int):
        train_data_files = [f"data/Protein/uniprot.ID/af2.files.secstruc.training.csv"]
        test_data_files = [f"data/Protein/uniprot.ID/af2.files.secstruc.testing.csv"]
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='pass',
                                                     tasks=['secondary_struct'],
                                                     repr_actions=['hidden_whole'],
                                                     freeze_language_model=True,
                                                     train_data_files=train_data_files,
                                                     test_data_files=test_data_files,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     device_id=device_id)
        epoch_losses = [[] for i in range(len(ids))]
        for epoch in range(epochs):
            start = time.time()
            losses = model_trainer.train_one_epoch()
            end = time.time()
            for i, loss in enumerate(losses):
                epoch_losses[i].append(loss)
            print(f"Epoch {epoch} loss: {np.mean(losses)}, time elapsed: {end - start}")
        n_cols = 1
        n_rows = np.ceil(len(ids) / n_cols).astype(int)
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_rows, 10 * n_cols))
        if n_cols > 1 and n_rows > 1:
            for i, gene in enumerate(ids):
                axs[i // n_cols, i % n_cols].plot(epoch_losses[i])
                axs[i // n_cols, i % n_cols].set_title(gene)
        else:
            axs.plot(epoch_losses[0])
            axs.set_title(ids[0])
        fig.savefig(f'{model_dir}/epoch.loss.pdf')
        return epoch_losses
    else:
        train_data_files = [f"data/Protein/uniprot.ID/af2.files.secstruc.training"]
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='pass',
                                                     tasks=['secondary_struct'],
                                                     repr_actions=['hidden_whole'],
                                                     freeze_language_model=True,
                                                     train_data_files=[None],
                                                     test_data_files=[None],
                                                     build_datasets=False,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     device_id=device_id)
        world_size = len(device_id)
        model_trainer.writer = None
        model_trainer.optimizer = None
        model_trainer.scheduler = None
        mp.spawn(
            data_distributed_parallel_gpu,
            args=(model_trainer, train_data_files, world_size, None, epochs, warmup_epochs),
            nprocs=world_size,
            join=True)
        return None


def _test_secondary_structures(ids,
                               language_model_name='esm1v',
                               model_dir='gMVP.style.esm.secstruc/',
                               device_id=1,
                               batch_size=None,
                               batch_number=5,
                               warmup_epochs=10,
                               epochs=30,
                               epoch_id=29,
                               counter_id=None):
    # check file existance
    result_dfs, label_dfs = [], []
    for gene in ids:
        if os.path.exists(f'{model_dir}/figs/{gene}.{epoch_id}.{counter_id}.csv'):
            print(f'Load precomputed {epoch_id} epoch result for {gene}')
            result_dfs.append(pd.read_csv(f'{model_dir}/figs/{gene}.{epoch_id}.{counter_id}.csv', index_col=0))
            if gene == "secstruc":
                label_df = pd.read_csv(f'{model_dir}/figs/{gene}.test.csv', index_col=0)
                label_dfs.append(label_df)
    os.makedirs(f'{model_dir}/figs/', exist_ok=True)
    if len(result_dfs) < len(ids):
        # Load pretrained model
        train_data_files = [f"data/Protein/uniprot.ID/af2.files.secstruc.training.csv"]
        test_data_files = [f"data/Protein/uniprot.ID/af2.files.secstruc.testing.csv"]
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='pass',
                                                     tasks=['secondary_struct'],
                                                     repr_actions=['hidden_whole'],
                                                     freeze_language_model=True,
                                                     train_data_files=train_data_files,
                                                     test_data_files=test_data_files,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     device_id=device_id)
        if counter_id is None:
            model_trainer.load_model(whole_model=True, epoch_id=epoch_id)
        else:
            model_trainer.load_model(whole_model=False, epoch_id=epoch_id, counter_id=counter_id)
        result_dfs, label_dfs = model_trainer.test_one_epoch(task_ids=[0])
    aucs = plot_aucs(ids, result_dfs, label_dfs, epoch_id, counter_id, model_dir)
    return aucs


def _test_secondary_structures_models(ids, language_model_name='esm1v',
                                      model_dir='gMVP.style.esm.secstruc/',
                                      device_id=1,
                                      batch_size=None,
                                      batch_number=5,
                                      epoch_ids=None,
                                      counter_ids_list=None):
    average_aucs = [[] for i in range(len(ids))]
    for epoch_id, counter_ids in zip(epoch_ids, counter_ids_list):
        for counter_id in counter_ids:
            average_auc = _test_secondary_structures(ids=ids,
                                                     language_model_name=language_model_name,
                                                     model_dir=model_dir,
                                                     device_id=device_id,
                                                     batch_size=batch_size,
                                                     batch_number=batch_number,
                                                     epoch_id=epoch_id,
                                                     counter_id=counter_id)
            print(f"Model epoch {epoch_id} counter {counter_id} average AUC: {np.mean(average_auc)}")
            for i, gene in enumerate(ids):
                average_aucs[i].append(average_auc[i])
    n_cols = 1
    n_rows = np.ceil(len(ids) / n_cols).astype(int)
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_cols, 10 * n_rows))
    if counter_ids_list is None or counter_ids_list[0][0] is None:
        axs.plot(epoch_ids, average_aucs[i])
    else:
        for i, gene in enumerate(ids):
            axs.plot(sum(counter_ids_list, []), average_aucs[i])
            axs.set_title(gene)
    fig.savefig(f'{model_dir}/figs/average.auc.from.{epoch_ids[0]}.to.{epoch_ids[-1]}.pdf')
    plt.close(fig)
    return average_aucs


def data_distributed_parallel_gpu(rank, trainer: GraphAttnVrtReprAgentTrainer,
                                  train_files,
                                  world_size, task_ids,
                                  epoches, warmup_epochs=5, step_scheduler=None):
    # set up training processes
    # Currently have bug if batch size does not match
    setup(rank, world_size)
    # zero the gradients before each epoch
    trainer.model.zero_grad()
    device = f'cuda:{rank}'
    torch.cuda.set_per_process_memory_fraction(1.0, rank)
    trainer.writer = SummaryWriter(log_dir=os.path.join(trainer.save_dir, "Log/"))
    # if task_ids is None, then train on all tasks
    if task_ids is None:
        task_ids = np.arange(trainer.task_number).tolist()
        tasks = trainer.task_number
    else:
        tasks = len(task_ids)
    # set up model to the correct device and set up ddp model
    model = trainer.model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=trainer.lr,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_epochs,
                                                num_training_steps=epoches,
                                                minlr=trainer.min_lr_ratio)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    trainer.build_train_datasets(train_data_files=[f"{train_file}.{rank}.csv" for train_file in train_files])
    if step_scheduler is not None:
        trainer.epoch_counter = step_scheduler
        for i in range(step_scheduler):
            scheduler.step()
        non_empty_train_sets = [trainer.train_data_sets[i].get_max_index() for i in range(trainer.task_number)
                                if trainer.train_data_sets[i] is not None]
        if len(non_empty_train_sets) > 0:
            trainer.writer_counter = step_scheduler * max(non_empty_train_sets)
    # all_losses = []
    for i in range(epoches):
        losses = [[] for k in range(tasks)]
        # TODO: Decide let all tasks run same batches or not
        # Trick here is, for one batch, we wait until all tasks has finished to calculate the gradient,
        # then we update the weights.
        task_finished = [False for k in range(tasks)]
        batch_count = 0
        # set up data sampler
        # sampler = [DistributedSampler(trainer.train_data_sets[i],
        #                               num_replicas=world_size,
        #                               rank=rank,
        #                               shuffle=True,  # May be True
        #                               seed=0) for i in task_ids]
        iter_data_loader = [iter(DataLoader(trainer.train_data_sets[i],
                                            batch_size=trainer.train_data_sets[i].batch_size,
                                            shuffle=False,  # Must be False!
                                            num_workers=10,
                                            # sampler=sampler[j],
                                            pin_memory=True)) for j, i in enumerate(task_ids)]
        while sum(task_finished) < tasks:
            batch_start_time = time.time()
            loss_sum = None
            for i, task_id in enumerate(task_ids):
                if task_finished[i]:
                    continue
                if trainer.tasks[task_id] == 'secondary_struct':
                    try:
                        batch_tokens, labels_ss, labels_rsa = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    batch_tokens = batch_tokens.to(device)
                    labels_ss = labels_ss.to(device)
                    labels_rsa = labels_rsa.to(device)
                    logits = ddp_model(batch_tokens, task_id=task_id)
                    loss_ss = trainer.loss_funcs[task_id][0](
                        logits[0].reshape(-1, logits[0].shape[-1]), labels_ss.reshape(-1)
                    )
                    loss_rsa = trainer.loss_funcs[task_id][1](logits[1][:, :, 0], labels_rsa)
                    trainer.write_loss([loss_ss.item(), loss_rsa.item()], task_id, rank_id=rank)
                    loss = loss_ss + loss_rsa
                else:
                    try:
                        batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, labels = next(iter_data_loader[i])
                    except StopIteration:
                        task_finished[i] = True
                        continue
                    batch_tokens_wt = batch_tokens_wt.to(device)
                    wt_aa = wt_aa.to(device)
                    vr_aa = vr_aa.to(device)
                    pos = pos.to(device)
                    msa_features = msa_features.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)
                    logits = ddp_model(batch_tokens_wt, wt_aa, vr_aa, pos, msa_features, masks, task_id=task_id)
                    if trainer.tasks[task_id] == 'ClinVar':
                        logits = logits[:, 0]
                        labels = labels.to(torch.float)
                    loss = trainer.loss_funcs[task_id](logits, labels)
                    trainer.write_loss(loss.item(), task_id, rank_id=rank)
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss
                losses[i].append(loss.item())
                dist.barrier()
            # step optimizer after all tasks have finished
            if loss_sum is None:
                break
            # add a sum of all parameters so that it will avoid stupid error of "unused parameter"
            loss_sum += 0 * sum([p.sum() for p in ddp_model.parameters()])
            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if rank == 0:
            trainer.writer_counter += 1
            batch_end_time = time.time()
            if rank == 0 and trainer.save_counters is not None and trainer.writer_counter % trainer.save_counters == 0:
                trainer.save_model(whole_model=False, counter_id=trainer.writer_counter)
            # wait for all ranks to update weights before next batch
            dist.barrier()
            print(f"Batch {batch_count} time: {batch_end_time - batch_start_time}")
            batch_count += 1
        # step scheduler after all batches have finished
        # only rank 0 updates the scheduler, epoch_counter
        scheduler.step()
        if rank == 0:
            trainer.epoch_counter += 1
            for i, task_id in enumerate(task_ids):
                print(f"Task {task_id} finished {len(losses[i])} batches with ",
                      f"loss: {np.mean(losses[i]):.4f}")
        if rank == 0 and trainer.epoch_counter % trainer.save_epochs == 0:
            trainer.save_model(whole_model=True)
        dist.barrier()
        # losses = [np.mean(losses[i]) for i in range(tasks)]
        # all_losses.append(losses)
    cleanup()
    # return all_losses


def train_clinVar(ids,
                  language_model_name='esm1v.secstruc',
                  model_dir='gMVP.style.esm.secstruc.ClinVar/',
                  device_id=1,
                  batch_size=None,
                  batch_number=5,
                  warmup_epochs=5,
                  epochs=20):
    # create output directory
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir), exist_ok=True)
    # Load pretrained model
    if device_id is None or isinstance(device_id, int):
        train_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/training.csv",
                            f"data/Protein/uniprot.ID/af2.files.secstruc.training.csv"]
        test_data_files = [None] * 2
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='attn',
                                                     tasks=['ClinVar', 'secondary_struct'],
                                                     repr_actions=['attn', 'hidden_whole'],
                                                     freeze_language_model=True,
                                                     train_data_files=train_data_files,
                                                     test_data_files=test_data_files,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     build_datasets=True,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     save_counters=50,
                                                     device_id=device_id)
        # model_trainer.load_model(whole_model=False, epoch_id=1, counter_id=2750)
        epoch_losses = [[] for i in range(len(ids))]
        for epoch in range(epochs):
            start = time.time()
            losses = model_trainer.train_one_epoch()
            end = time.time()
            for i, loss in enumerate(losses):
                epoch_losses[i].append(loss)
            print(f"Epoch {epoch} loss: {np.mean(losses)}, time elapsed: {end - start}")
        n_cols = 1
        n_rows = np.ceil(len(ids) / n_cols).astype(int)
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_rows, 10 * n_cols))
        if n_cols > 1 and n_rows > 1:
            for i, gene in enumerate(ids):
                axs[i // n_cols, i % n_cols].plot(epoch_losses[i])
                axs[i // n_cols, i % n_cols].set_title(gene)
        else:
            axs.plot(epoch_losses[0])
            axs.set_title(ids[0])
        fig.savefig(f'{model_dir}/epoch.loss.pdf')
        return epoch_losses
    else:
        train_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/training",
                            f"data/Protein/uniprot.ID/af2.files.secstruc.training"]
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='attn',
                                                     tasks=['ClinVar', 'secondary_struct'],
                                                     repr_actions=['attn', 'hidden_whole'],
                                                     freeze_language_model=True,
                                                     train_data_files=[None] * 2,
                                                     test_data_files=[None] * 2,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     build_datasets=False,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     save_counters=50,
                                                     device_id=device_id)
        # model_trainer.load_model(whole_model=True, epoch_id=10, counter_id=None)
        world_size = len(device_id)
        model_trainer.writer = None
        model_trainer.optimizer = None
        model_trainer.scheduler = None
        mp.spawn(
            data_distributed_parallel_gpu,
            args=(model_trainer, train_data_files, world_size, None, epochs, warmup_epochs),
            nprocs=world_size,
            join=True)
        return None


def load_precomputed_model(model_dir, ids, epoch_id, counter_id):
    loaded_result_dfs, loaded_label_dfs, to_compute = [], [], []
    for gene in ids:
        if os.path.exists(f'{model_dir}/figs/{gene}.{epoch_id}.{counter_id}.csv'):
            print(f'Load precomputed {epoch_id} epoch result for {gene}')
            loaded_result_dfs.append(pd.read_csv(f'{model_dir}/figs/{gene}.{epoch_id}.{counter_id}.csv', index_col=0))
            if gene == "ClinVar":
                label_df = pd.read_csv(f"data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv",
                                       index_col=0)
                loaded_label_dfs.append(label_df[['score']])
            elif gene == "secstruc":
                label_df = pd.read_csv(f'data/Protein/uniprot.ID/secstruc.test.csv',
                                       index_col=0)
                loaded_label_dfs.append(label_df)
            elif gene.startswith('PF'):
                label_df = pd.read_csv(f'data/DMS/Itan_AJHG/pfams.0.8/{gene}/testing.csv',
                                       index_col=0)
                loaded_label_dfs.append(label_df[['score']])
            else:
                label_df = pd.read_csv(f'data/DMS/MAVEDB/{gene}/testing.csv',
                                       index_col=0)
                loaded_label_dfs.append(label_df.loc[:, [i for i in label_df.columns if i.startswith('score')]])
            to_compute.append(False)
        else:
            to_compute.append(True)
    return loaded_result_dfs, loaded_label_dfs, to_compute


def combine_loaded_computed_results(loaded_result_dfs, loaded_label_dfs,
                                    to_compute, computed_result_dfs, computed_label_dfs):
    result_dfs, label_dfs = [], []
    iter_computed_result_dfs = iter(computed_result_dfs)
    iter_computed_label_dfs = iter(computed_label_dfs)
    iter_loaded_result_dfs = iter(loaded_result_dfs)
    iter_loaded_label_dfs = iter(loaded_label_dfs)
    for i in to_compute:
        if i:
            result_dfs.append(next(iter_computed_result_dfs))
            label_dfs.append(next(iter_computed_label_dfs))
        else:
            result_dfs.append(next(iter_loaded_result_dfs))
            label_dfs.append(next(iter_loaded_label_dfs))
    return result_dfs, label_dfs


def _test_clinVar(ids,
                  language_model_name='esm1v.secstruc',
                  model_dir='gMVP.style.esm.secstruc.ClinVar/',
                  device_id=1,
                  batch_size=None,
                  batch_number=5,
                  warmup_epochs=10,
                  epochs=50,
                  epoch_id=49,
                  counter_id=None):
    # check file existance
    loaded_result_dfs, loaded_label_dfs, to_compute = load_precomputed_model(model_dir, ids, epoch_id, counter_id)
    if len(loaded_result_dfs) < len(ids):
        # read msa files
        train_data_files = [None] * len(ids)
        train_data_files = [None if not to_compute[i] else train_data_files[i] for i in range(len(ids))]
        test_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv",
                           f"data/Protein/uniprot.ID/af2.files.secstruc.testing.csv"]
        test_data_files = [None if not to_compute[i] else test_data_files[i] for i in range(len(ids))]
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='attn',
                                                     tasks=['ClinVar', 'secondary_struct'],
                                                     repr_actions=['attn', 'hidden_whole'],
                                                     freeze_language_model=True,
                                                     train_data_files=train_data_files,
                                                     test_data_files=test_data_files,
                                                     build_datasets=True,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     device_id=device_id)
        if counter_id is None:
            model_trainer.load_model(whole_model=True, epoch_id=epoch_id)
        else:
            model_trainer.load_model(whole_model=False, epoch_id=epoch_id, counter_id=counter_id)
        computed_result_dfs, computed_label_dfs = \
            model_trainer.test_one_epoch(task_ids=np.where(to_compute)[0].tolist())
    else:
        computed_result_dfs, computed_label_dfs = [], []
    result_dfs, label_dfs = combine_loaded_computed_results(loaded_result_dfs, loaded_label_dfs,
                                                            to_compute, computed_result_dfs, computed_label_dfs)
    aucs = plot_aucs(ids, result_dfs, label_dfs, epoch_id, counter_id, model_dir)
    return aucs


def _get_embedding_DMS(ids,
                       language_model_name='esm1v.secstruc',
                       model_dir='gMVP.style.esm.secstruc.ClinVar/',
                       device_id=1,
                       task_out_dim=None,
                       batch_size=None,
                       batch_number=5,
                       warmup_epochs=10,
                       epochs=50,
                       epoch_id=49,
                       counter_id=None):
    # currently only support one gene one time
    ids = ['ClinVar'] + ids
    # read msa files
    train_data_files = [None] * len(ids)
    test_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv"] + \
                      [f"data/DMS/MAVEDB/{gene}/assay.csv"
                       for gene in ids[1:]]
    model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                 language_model_repr_layer=33,
                                                 hidden_layer='attn',
                                                 task_out_dim=[1] + task_out_dim,
                                                 # tasks=['ClinVar', 'secondary_struct'],
                                                 # repr_actions=['attn', 'hidden_whole'],
                                                 tasks=['ClinVar'] + ['DMS'] * len(ids[1:]),
                                                 repr_actions=['attn'] + ['attn'] * len(ids[1:]),
                                                 freeze_language_model=True,
                                                 train_data_files=train_data_files,
                                                 test_data_files=test_data_files,
                                                 # build_datasets=False,
                                                 save_dir=model_dir,
                                                 batch_sizes=batch_size,
                                                 batch_numbers=batch_number,
                                                 num_warmup_epochs=warmup_epochs,
                                                 num_training_epochs=epochs,
                                                 device_id=device_id)
    if counter_id is None:
        model_trainer.load_model(whole_model=True, epoch_id=epoch_id)
    else:
        model_trainer.load_model(whole_model=False, epoch_id=epoch_id, counter_id=counter_id)
    computed_result_dfs, _ = model_trainer.get_embedding_one_epoch(task_ids=[1])
    for i, gene in enumerate(ids[1:]):
        computed_result_dfs[i].to_csv(f'{model_dir}/figs/{gene}.{epoch_id}.{counter_id}.embeddings.csv')
    return computed_result_dfs


def _test_clinVar_models(ids,
                         language_model_name='esm1v.secstruc',
                         model_dir='gMVP.style.esm.secstruc.ClinVar/',
                         epoch_ids=None,
                         counter_ids_list=None,
                         batch_size=None,
                         batch_number=5,
                         device_id=0):
    os.makedirs(f'{model_dir}/figs', exist_ok=True)
    average_aucs = [[] for i in range(len(ids))]
    for epoch_id, counter_ids in zip(epoch_ids, counter_ids_list):
        for counter_id in counter_ids:
            average_auc = _test_clinVar(ids=ids,
                                        language_model_name=language_model_name,
                                        model_dir=model_dir,
                                        epoch_id=epoch_id,
                                        counter_id=counter_id,
                                        batch_size=batch_size,
                                        batch_number=batch_number,
                                        device_id=device_id)
            print(f"Model epoch {epoch_id} counter {counter_id} average AUC: {np.mean(average_auc)}")
            for i, gene in enumerate(ids):
                average_aucs[i].append(average_auc[i])
    n_cols = 2
    n_rows = np.ceil(len(ids) / n_cols).astype(int)
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_cols, 10 * n_rows))
    for i, gene in enumerate(ids):
        if counter_ids_list[0][0] is None:
            axs[i % n_cols].plot(epoch_ids, average_aucs[i], label=gene)
        else:
            axs[i % n_cols].plot(sum(counter_ids_list, []), average_aucs[i])
        # axs[i % n_cols].set_title(gene)
    fig.savefig(f'{model_dir}/figs/average.auc.from.{epoch_ids[0]}.to.{epoch_ids[-1]}.pdf')
    return average_aucs


def train_itan(ids,
               language_model_name='esm1v.secstruc.ClinVar',
               model_dir='gMVP.style.esm.secstruc.ClinVar.itan/',
               device_id=1,
               batch_size=None,
               batch_number=5,
               warmup_epochs=5,
               epochs=20,
               seed=0):
    # create output directory
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir), exist_ok=True)
    # Load pretrained model and data
    train_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/training.fine.tune.csv"] + \
                       [f"data/DMS/Itan_AJHG/pfams.0.8/{gene}/training.csv"
                        for gene in ids]
    test_data_files = [None] * (1 + len(ids))
    model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                 language_model_repr_layer=33,
                                                 hidden_layer='attn',
                                                 tasks=['ClinVar'] + ['GoF_LoF'] * len(ids),
                                                 repr_actions=['attn'] + ['attn'] * len(ids),
                                                 freeze_language_model=True,
                                                 train_data_files=train_data_files,
                                                 test_data_files=test_data_files,
                                                 save_dir=model_dir,
                                                 batch_sizes=batch_size,
                                                 batch_numbers=batch_number,
                                                 build_datasets=True,
                                                 task_out_dim=[1] + [2] * len(ids),
                                                 lr=5e-6,
                                                 num_warmup_epochs=warmup_epochs,
                                                 num_training_epochs=epochs,
                                                 save_counters=50,
                                                 device_id=device_id,
                                                 seed=seed)
    # model_trainer.load_model(whole_model=False, epoch_id=1, counter_id=2750)
    ids = ['ClinVar'] + ids
    epoch_losses = [[] for i in range(len(ids))]
    for epoch in range(epochs):
        start = time.time()
        losses = model_trainer.fine_tune_one_epoch(fine_tune_task_ids=list(range(1, 1 + len(ids))))
        end = time.time()
        for i, loss in enumerate(losses):
            epoch_losses[i].append(loss)
        print(f"Epoch {epoch} loss: {np.mean(losses)}, time elapsed: {end - start}")
    plot_loss(ids, epoch_losses, model_dir)
    return epoch_losses


def train_itan_no_pretrain(ids,
                           language_model_name='esm1v.secstruc.ClinVar',
                           model_dir='gMVP.style.esm.secstruc.ClinVar.itan/',
                           device_id=1,
                           batch_size=None,
                           batch_number=5,
                           warmup_epochs=5,
                           epochs=20,
                           seed=0):
    # create output directory
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir), exist_ok=True)
    # Load pretrained model and data
    train_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/training.fine.tune.csv"] + \
                       [f"data/DMS/Itan_AJHG/pfams.0.8/{gene}/training.csv"
                        for gene in ids]
    test_data_files = [None] * (1 + len(ids))
    model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                 language_model_repr_layer=33,
                                                 hidden_layer='attn',
                                                 tasks=['ClinVar'] + ['GoF_LoF'] * len(ids),
                                                 repr_actions=['attn'] + ['attn'] * len(ids),
                                                 freeze_language_model=True,
                                                 train_data_files=train_data_files,
                                                 test_data_files=test_data_files,
                                                 save_dir=model_dir,
                                                 batch_sizes=batch_size,
                                                 batch_numbers=batch_number,
                                                 build_datasets=True,
                                                 task_out_dim=[1] + [2] * len(ids),
                                                 lr=5e-6,
                                                 num_warmup_epochs=warmup_epochs,
                                                 num_training_epochs=epochs,
                                                 save_counters=50,
                                                 device_id=device_id,
                                                 seed=seed)
    # model_trainer.load_model(whole_model=False, epoch_id=1, counter_id=2750)
    ids = ['ClinVar'] + ids
    epoch_losses = [[] for i in range(len(ids))]
    for epoch in range(epochs):
        start = time.time()
        losses = model_trainer.fine_tune_no_pretrain_one_epoch(fine_tune_task_ids=list(range(1, 1 + len(ids))))
        end = time.time()
        for i, loss in enumerate(losses):
            epoch_losses[i].append(loss)
        print(f"Epoch {epoch} loss: {np.mean(losses)}, time elapsed: {end - start}")
    plot_loss(ids, epoch_losses, model_dir)
    return epoch_losses


def train_DMS(ids,
              language_model_name='esm1v.secstruc.ClinVar',
              model_dir='gMVP.style.esm.secstruc.ClinVar.DMS/',
              task_out_dim=None,
              device_id=1,
              batch_size=None,
              batch_number=5,
              warmup_epochs=5,
              epochs=20,
              seed=0):
    # create output directory
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir), exist_ok=True)
    # Load pretrained model and data
    train_data_files = [
                           f"data/DMS/ClinVar.HGMD.PrimateAI.syn/training.fine.tune.csv"] + \
                       [f"data/DMS/MAVEDB/{gene}/training.csv"
                        for gene in ids]
    test_data_files = [None] * (1 + len(ids))
    model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                 language_model_repr_layer=33,
                                                 hidden_layer='attn',
                                                 tasks=['ClinVar'] + ['DMS'] * len(ids),
                                                 task_out_dim=[1] + task_out_dim,
                                                 repr_actions=['attn'] + ['attn'] * len(ids),
                                                 freeze_language_model=True,
                                                 train_data_files=train_data_files,
                                                 test_data_files=test_data_files,
                                                 save_dir=model_dir,
                                                 batch_sizes=batch_size,
                                                 batch_numbers=batch_number,
                                                 build_datasets=True,
                                                 num_warmup_epochs=warmup_epochs,
                                                 num_training_epochs=epochs,
                                                 save_counters=50,
                                                 device_id=device_id,
                                                 seed=seed)
    # model_trainer.load_model(whole_model=False, epoch_id=1, counter_id=2750)
    ids = ['ClinVar'] + ids
    epoch_losses = [[] for i in range(len(ids) + 1)]
    for epoch in range(epochs):
        start = time.time()
        losses = model_trainer.fine_tune_one_epoch(fine_tune_task_ids=list(range(1, 1 + len(ids))))
        end = time.time()
        for i, loss in enumerate(losses):
            epoch_losses[i].append(loss)
        print(f"Epoch {epoch} loss: {np.mean(losses)}, time elapsed: {end - start}")
    plot_loss(ids, epoch_losses, model_dir)
    return epoch_losses


def train_DMS_no_pretrain(ids,
                          language_model_name='esm1v.secstruc.ClinVar',
                          model_dir='gMVP.style.esm.secstruc.ClinVar.DMS/',
                          task_out_dim=None,
                          device_id=1,
                          batch_size=None,
                          batch_number=5,
                          warmup_epochs=5,
                          epochs=20,
                          seed=0):
    # create output directory
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir), exist_ok=True)
    # Load pretrained model and data
    train_data_files = [
                           f"data/DMS/ClinVar.HGMD.PrimateAI.syn/training.fine.tune.csv"] + \
                       [f"data/DMS/MAVEDB/{gene}/training.csv"
                        for gene in ids]
    test_data_files = [None] * (1 + len(ids))
    model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                 language_model_repr_layer=33,
                                                 hidden_layer='attn',
                                                 tasks=['ClinVar'] + ['DMS'] * len(ids),
                                                 task_out_dim=[1] + task_out_dim,
                                                 repr_actions=['attn'] + ['attn'] * len(ids),
                                                 freeze_language_model=True,
                                                 train_data_files=train_data_files,
                                                 test_data_files=test_data_files,
                                                 save_dir=model_dir,
                                                 batch_sizes=batch_size,
                                                 batch_numbers=batch_number,
                                                 build_datasets=True,
                                                 num_warmup_epochs=warmup_epochs,
                                                 num_training_epochs=epochs,
                                                 save_counters=50,
                                                 device_id=device_id,
                                                 seed=seed)
    # model_trainer.load_model(whole_model=False, epoch_id=1, counter_id=2750)
    ids = ['ClinVar'] + ids
    epoch_losses = [[] for i in range(len(ids) + 1)]
    for epoch in range(epochs):
        start = time.time()
        losses = model_trainer.fine_tune_no_pretrain_one_epoch(fine_tune_task_ids=list(range(1, 1 + len(ids))))
        end = time.time()
        for i, loss in enumerate(losses):
            epoch_losses[i].append(loss)
        print(f"Epoch {epoch} loss: {np.mean(losses)}, time elapsed: {end - start}")
    plot_loss(ids, epoch_losses, model_dir)
    return epoch_losses


def _test_itan(ids,
               language_model_name='esm1v.secstruc',
               model_dir='gMVP.style.esm.secstruc.ClinVar/',
               device_id=1,
               task_out_dim=None,
               batch_size=None,
               batch_number=5,
               warmup_epochs=10,
               epochs=50,
               epoch_id=49,
               counter_id=None):
    # check file existance
    ids = ['ClinVar'] + ids
    loaded_result_dfs, loaded_label_dfs, to_compute = load_precomputed_model(model_dir, ids, epoch_id, counter_id)
    if len(loaded_result_dfs) < len(ids):
        # read msa files
        train_data_files = [None] * len(ids)
        test_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv"] + \
                          [f"data/DMS/Itan_AJHG/pfams.0.8/{gene}/testing.csv"
                           for gene in ids[1:]]
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='attn',
                                                     tasks=['ClinVar'] + ['GoF_LoF'] * len(ids[1:]),
                                                     repr_actions=['attn'] + ['attn'] * len(ids[1:]),
                                                     freeze_language_model=True,
                                                     train_data_files=train_data_files,
                                                     test_data_files=test_data_files,
                                                     # build_datasets=False,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     device_id=device_id)
        if counter_id is None:
            model_trainer.load_model(whole_model=True, epoch_id=epoch_id)
        else:
            model_trainer.load_model(whole_model=False, epoch_id=epoch_id, counter_id=counter_id)
        computed_result_dfs, computed_label_dfs = \
            model_trainer.test_one_epoch(task_ids=np.where(to_compute)[0].tolist())
    else:
        computed_result_dfs, computed_label_dfs = [], []
    result_dfs, label_dfs = combine_loaded_computed_results(loaded_result_dfs, loaded_label_dfs,
                                                            to_compute, computed_result_dfs, computed_label_dfs)
    aucs = plot_aucs(ids, result_dfs, label_dfs, epoch_id, counter_id, model_dir)
    return aucs


def _test_DMS(ids,
              language_model_name='esm1v.secstruc',
              model_dir='gMVP.style.esm.secstruc.ClinVar/',
              device_id=1,
              task_out_dim=None,
              batch_size=None,
              batch_number=5,
              warmup_epochs=10,
              epochs=50,
              epoch_id=49,
              counter_id=None):
    # check file existance
    ids = ['ClinVar'] + ids
    loaded_result_dfs, loaded_label_dfs, to_compute = load_precomputed_model(model_dir, ids, epoch_id, counter_id)
    if len(loaded_result_dfs) < len(ids):
        # read msa files
        train_data_files = [None] * len(ids)
        test_data_files = [f"data/DMS/ClinVar.HGMD.PrimateAI.syn/testing.csv"] + \
                          [f"data/DMS/MAVEDB/{gene}/testing.csv"
                           for gene in ids[1:]]
        model_trainer = GraphAttnVrtReprAgentTrainer(language_model_name=language_model_name,
                                                     language_model_repr_layer=33,
                                                     hidden_layer='attn',
                                                     task_out_dim=[1] + task_out_dim,
                                                     tasks=['ClinVar'] + ['DMS'] * len(ids[1:]),
                                                     repr_actions=['attn'] + ['attn'] * len(ids[1:]),
                                                     freeze_language_model=True,
                                                     train_data_files=train_data_files,
                                                     test_data_files=test_data_files,
                                                     # build_datasets=False,
                                                     save_dir=model_dir,
                                                     batch_sizes=batch_size,
                                                     batch_numbers=batch_number,
                                                     num_warmup_epochs=warmup_epochs,
                                                     num_training_epochs=epochs,
                                                     device_id=device_id)
        if counter_id is None:
            model_trainer.load_model(whole_model=True, epoch_id=epoch_id)
        else:
            model_trainer.load_model(whole_model=False, epoch_id=epoch_id, counter_id=counter_id)
        computed_result_dfs, computed_label_dfs = \
            model_trainer.test_one_epoch(task_ids=np.where(to_compute)[0].tolist())
    else:
        computed_result_dfs, computed_label_dfs = [], []
    result_dfs, label_dfs = combine_loaded_computed_results(loaded_result_dfs, loaded_label_dfs,
                                                            to_compute, computed_result_dfs, computed_label_dfs)
    aucs = plot_aucs(ids, result_dfs, label_dfs, epoch_id, counter_id, model_dir)
    return aucs


def _test_models(ids,
                 _test_fn,
                 language_model_name='esm1v.secstruc',
                 model_dir='gMVP.style.esm.secstruc.ClinVar/',
                 task_out_dim=None,
                 epoch_ids=None,
                 counter_ids_list=None,
                 batch_size=None,
                 batch_number=5,
                 device_id=0):
    if _test_fn == _test_DMS or _test_fn == _test_itan or _test_fn == _get_embedding_DMS:
        prepended_ids = ['ClinVar'] + ids
        average_aucs = [[] for _ in range(len(prepended_ids))]
    else:
        prepended_ids = None
        average_aucs = [[] for _ in range(len(ids))]
    os.makedirs(f"{model_dir}/figs/", exist_ok=True)
    for epoch_id, counter_ids in zip(epoch_ids, counter_ids_list):
        for counter_id in counter_ids:
            average_auc = _test_fn(ids=ids,
                                   language_model_name=language_model_name,
                                   model_dir=model_dir,
                                   task_out_dim=task_out_dim,
                                   epoch_id=epoch_id,
                                   counter_id=counter_id,
                                   batch_size=batch_size,
                                   batch_number=batch_number,
                                   device_id=device_id)
            if _test_fn != _get_embedding_DMS:
                print(f"Model epoch {epoch_id} counter {counter_id} average AUC: {np.mean(average_auc)}")
                for i, auc in enumerate(average_auc):
                    average_aucs[i].append(auc)
    if _test_fn != _get_embedding_DMS:
        if prepended_ids is not None:
            ids = prepended_ids
        n_cols = 2
        n_rows = np.ceil(len(ids) / n_cols).astype(int)
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(10 * n_cols, 10 * n_rows))
        for i, gene in enumerate(ids):
            if n_rows > 1:
                if counter_ids_list[0][0] is None:
                    axs[i // n_cols, i % n_cols].plot(epoch_ids, average_aucs[i], label=gene)
                else:
                    axs[i // n_cols, i % n_cols].plot(sum(counter_ids_list, []), average_aucs[i])
                axs[i // n_cols, i % n_cols].set_title(gene)
            else:
                if counter_ids_list[0][0] is None:
                    axs[i % n_cols].plot(epoch_ids, average_aucs[i], label=gene)
                else:
                    axs[i % n_cols].plot(sum(counter_ids_list, []), average_aucs[i])
                axs[i % n_cols].set_title(gene)
        fig.savefig(f'{model_dir}/figs/average.auc.from.{epoch_ids[0]}.to.{epoch_ids[-1]}.pdf')
    return average_aucs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    # Train esm model on secondary structures with only a MLP layer
    if args.mode == 0:
        language_model_name = 'esm1b'
        ids = ['secstruc']
        model_dir = f'model_checkpoints/gMVP.style.esm1b.pass.secstruc/'
        batch_size = 130
        batch_number = None
        losses = train_secondary_structures(ids=ids,
                                            language_model_name=language_model_name,
                                            batch_size=batch_size,
                                            batch_number=batch_number,
                                            model_dir=model_dir,
                                            device_id=[0, 1, 2])
    # Test esm model on secondary structures with only a MLP layer
    elif args.mode == 1:
        language_model_name = 'esm2'
        ids = ['secstruc']
        model_dir = f'model_checkpoints/gMVP.style.esm2.pass.secstruc/'
        batch_size = 130
        batch_number = None
        losses = _test_secondary_structures_models(ids=ids,
                                                   language_model_name=language_model_name,
                                                   batch_size=batch_size,
                                                   batch_number=batch_number,
                                                   model_dir=model_dir,
                                                   device_id=1,
                                                   # epoch_ids=[1, 2, 4, 6, 8, 10, 14, 18, 22, 26, 29],
                                                   epoch_ids=[30],
                                                   counter_ids_list=[[None]])
    # Pretrain RESCVE model on pathogenicity and secondary structures
    elif args.mode == 2:
        language_model_name = 'esm1b'
        ids = ['ClinVar', 'secstruc']
        model_dir = f'model_checkpoints/gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5/'
        batch_size = [95, 18]
        batch_number = [None] * 2
        losses = train_clinVar(ids=ids,
                               language_model_name=language_model_name,
                               batch_size=batch_size,
                               batch_number=batch_number,
                               model_dir=model_dir,
                               device_id=[0, 1, 2])
    # Test RESCVE model on pathogenicity and secondary structures
    elif args.mode == 3:
        language_model_name = 'esm2'
        ids = ['ClinVar', 'secstruc']
        model_dir = f'model_checkpoints/gMVP.style.esm2.secstruc.CHPs.all.transformerLayer5/'
        batch_size = [100, 50]
        batch_number = [None] * 2
        device_id = 3
        # epoch_ids = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 45, 50]
        # epoch_ids = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
        epoch_ids = [20]
        counter_ids_list = [[None]] * len(epoch_ids)
        # epoch_ids = [4]
        # counter_ids_list = [[2050]]
        losses = _test_clinVar_models(ids=ids,
                                      language_model_name=language_model_name,
                                      batch_size=batch_size,
                                      batch_number=batch_number,
                                      model_dir=model_dir,
                                      epoch_ids=epoch_ids,
                                      counter_ids_list=counter_ids_list,
                                      device_id=device_id)
    # Transfer learning of RESCVE model on PTEN dataset
    elif args.mode == 4:
        language_model_name = 'esm1b.secstruc.CHPs'
        ids = ['PTEN']
        seed = args.seed
        model_dir = f'model_checkpoints/gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.PTEN.seed.{seed}/'
        batch_size = [30] + [30]
        batch_number = [None] * (1 + 1)
        task_out_dim = [2]
        device_id = args.device
        losses = train_DMS(ids=ids,
                           language_model_name=language_model_name,
                           batch_size=batch_size,
                           task_out_dim=task_out_dim,
                           batch_number=batch_number,
                           model_dir=model_dir,
                           device_id=device_id)
        epoch_ids = [20]
        counter_ids_list = [[None]] * len(epoch_ids)
        losses = _test_models(ids=ids,
                              _test_fn=_test_DMS,
                              language_model_name=language_model_name,
                              batch_size=batch_size,
                              task_out_dim=task_out_dim,
                              batch_number=batch_number,
                              model_dir=model_dir,
                              epoch_ids=epoch_ids,
                              counter_ids_list=counter_ids_list,
                              device_id=device_id)
    # Transfer learning of RESCVE model on Protein Family specific datasets
    # Note here we have two options, one is ion channel (larger), one is other protein families.
    # To test one of those, uncomment the lines.
    elif args.mode == 5:
        language_model_name = 'esm1b.secstruc.CHPs'
        # ids = ['PF07714', 'PF00027', 'PF00017']
        ids = ['PF.ion.channel']
        seed = args.seed
        model_dir = f'model_checkpoints/gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.ion.channel.seed.{seed}/'
        batch_size = [10] * (1 + len(ids))
        batch_number = [None] * (1 + len(ids))
        device_id = args.device
        losses = train_itan(ids=ids,
                            language_model_name=language_model_name,
                            batch_size=batch_size,
                            batch_number=batch_number,
                            model_dir=model_dir,
                            device_id=device_id,
                            seed=seed)
        epoch_ids = [20]
        counter_ids_list = [[None]] * len(epoch_ids)
        losses = _test_models(ids=ids,
                              _test_fn=_test_itan,
                              language_model_name=language_model_name,
                              batch_size=batch_size,
                              task_out_dim=None,
                              batch_number=batch_number,
                              model_dir=model_dir,
                              epoch_ids=epoch_ids,
                              counter_ids_list=counter_ids_list,
                              device_id=device_id)
    # Transfer learning of un-pre-trained RESCVE model on PTEN dataset
    elif args.mode == 6:
        language_model_name = 'esm1b'
        ids = ['PTEN']
        seed = args.seed
        model_dir = f'model_checkpoints/gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.PTEN.seed.{seed}/'
        batch_size = [30] + [30]
        batch_number = [None] * (1 + 1)
        task_out_dim = [2]
        device_id = args.device
        losses = train_DMS(ids=ids,
                           language_model_name=language_model_name,
                           batch_size=batch_size,
                           task_out_dim=task_out_dim,
                           batch_number=batch_number,
                           model_dir=model_dir,
                           device_id=device_id)
        epoch_ids = [20]
        counter_ids_list = [[None]] * len(epoch_ids)
        losses = _test_models(ids=ids,
                              _test_fn=_test_DMS,
                              language_model_name=language_model_name,
                              batch_size=batch_size,
                              task_out_dim=task_out_dim,
                              batch_number=batch_number,
                              model_dir=model_dir,
                              epoch_ids=epoch_ids,
                              counter_ids_list=counter_ids_list,
                              device_id=device_id)
    # Transfer learning of un-pre-trained RESCVE model on Protein Family specific datasets
    elif args.mode == 7:
        language_model_name = 'esm1b'
        # ids = ['PF07714', 'PF00027', 'PF00017']
        ids = ['PF.ion.channel']
        seed = args.seed
        model_dir = f'model_checkpoints/gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.ion.channel.seed.{seed}/'
        batch_size = [10] * (1 + len(ids))
        batch_number = [None] * (1 + len(ids))
        device_id = args.device
        losses = train_itan_no_pretrain(ids=ids,
                                        language_model_name=language_model_name,
                                        batch_size=batch_size,
                                        batch_number=batch_number,
                                        model_dir=model_dir,
                                        device_id=device_id,
                                        seed=seed)
        epoch_ids = [20]
        counter_ids_list = [[None]] * len(epoch_ids)
        losses = _test_models(ids=ids,
                              _test_fn=_test_itan,
                              language_model_name=language_model_name,
                              batch_size=batch_size,
                              task_out_dim=None,
                              batch_number=batch_number,
                              model_dir=model_dir,
                              epoch_ids=epoch_ids,
                              counter_ids_list=counter_ids_list,
                              device_id=device_id)
    # Test of transfer learning result on PTEN
    elif args.mode == 8:
        language_model_name = 'esm1b.secstruc.CHPs'
        ids = ['PTEN']
        model_dir = f'model_checkpoints/gMVP.style.esm1b.secstruc.CHPs.all.transformerLayer5.PTEN/'
        batch_size = [60] * 2
        batch_number = [None] * (1 + 1)
        task_out_dim = [2]
        device_id = 3
        # epoch_ids = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 45, 50]
        # epoch_ids = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 18, 20]
        epoch_ids = [20]
        counter_ids_list = [[None]] * len(epoch_ids)
        losses = _test_models(ids=ids,
                              _test_fn=_test_DMS,
                              language_model_name=language_model_name,
                              batch_size=batch_size,
                              task_out_dim=task_out_dim,
                              batch_number=batch_number,
                              model_dir=model_dir,
                              epoch_ids=epoch_ids,
                              counter_ids_list=counter_ids_list,
                              device_id=device_id)
    # Test of transfer learning result on Protein Family specific datasets
    elif args.mode == 9:
        language_model_name = 'esm1b.secstruc'
        # ids = ['PF07714', 'PF00027', 'PF00017']
        ids = ['PF.ion.channel']
        model_dir = f'model_checkpoints/gMVP.style.esm1b.transformerLayer5.ion.channel/'
        batch_size = [20] * (1 + len(ids))
        batch_number = [None] * (1 + len(ids))
        device_id = 3
        epoch_ids = [20]
        counter_ids_list = [[None]] * len(epoch_ids)
        losses = _test_models(ids=ids,
                              _test_fn=_test_itan,
                              language_model_name=language_model_name,
                              batch_size=batch_size,
                              task_out_dim=None,
                              batch_number=batch_number,
                              model_dir=model_dir,
                              epoch_ids=epoch_ids,
                              counter_ids_list=counter_ids_list,
                              device_id=device_id)
