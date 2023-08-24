# !pip install transformers
# !pip install datasets
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from os.path import dirname, abspath, join
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
# sys.path.append(join(PROJECT_ROOT, 'src'))
sys.path.append(PROJECT_ROOT)


import random
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    np.random.seed((torch_seed + worker_id) % 2**30)


mitsar_chls = ['Fp1', 'Fp2', 'FZ', 'FCz', 'Cz', 'Pz', 'O1', 'O2', 'F3', 'F4',
               'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4', 'T5', 'T6', 'A1', 'A2']
mitsar_chls = [i.upper() for i in mitsar_chls]

HSE_chls = ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6',
    'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2']

HSE_chls = [i.upper() for i in HSE_chls]


class TEST_v7(torch.utils.data.Dataset):
    def __init__(self, main, labels, norm, chls, correct):
        super(TEST_v7, self).__init__()
        self.main = main
        self.label = labels
        self.norm = norm
        self.chls = chls
        self.correct = correct

    def __len__(self):
        return len(self.main)

    def __getitem__(self, idx):
        # sample = torch.from_numpy(np.load(self.meta.iloc[idx]['path'])[:6000].astype(np.float32)).clone()
        sample = np.copy(self.main[idx])
        # sample = butter_bandpass_filter_v2(sample, 1, 40, 100)
        # sample_label = torch.tensor(np.copy(self.label[idx]))-1#torch.tensor(1 if self.main[idx]['label'] == 'work' else 0)
        sample_label = torch.tensor(0) if self.label[idx] == 0 else torch.tensor(1)

        if self.label[idx] == 0:
            sample_label = torch.tensor(0)
        elif self.label[idx] == 1:
            sample_label = torch.tensor(1)
        elif self.label[idx] == 2:
            sample_label = torch.tensor(2)
        elif self.label[idx] == 3:
            sample_label = torch.tensor(3)

        sample_mean_std = self.norm[idx]
        try:
            sample_correct = torch.tensor(self.correct[idx])
        except:
            sample_correct = torch.tensor(-1)
        # sample_label = label_map[sample['label']]

        channels = [mitsar_chls.index('FP1'), mitsar_chls.index('FP2')]
        sample = sample[:, [self.chls.index('FP1'), self.chls.index('FP2')]]
        # sample_min, sample_max = sample.mean(0), sample.std(0)
        # sample = (sample - sample_min) / (sample_max - sample_min)
        sample[:, 0] -= sample_mean_std[0][self.chls.index('FP1')]
        sample[:, 1] -= sample_mean_std[0][self.chls.index('FP2')]
        sample[:, 0] /= sample_mean_std[1][self.chls.index('FP1')]
        sample[:, 1] /= sample_mean_std[1][self.chls.index('FP2')]

        # sample = butter_bandpass_filter_v2(sample, 1, 40, 256)
        sample = torch.from_numpy(sample[:3000].astype(np.float32)).clone()
        return {'anchor': sample,
                'label': sample_label,
                'channels': torch.tensor(channels),
                'correct': sample_correct,
                'pure_label': torch.tensor(self.label[idx])}

class TEST(torch.utils.data.Dataset):
  def __init__(self, main, labels, norm):
    super(TEST, self).__init__()
    self.main = main
    self.label = labels
    self.norm = norm

  def __len__(self):
    return len(self.main)

  def __getitem__(self, idx):
    # sample = torch.from_numpy(np.load(self.meta.iloc[idx]['path'])[:6000].astype(np.float32)).clone()
    sample = np.copy(self.main[idx])
    # sample = butter_bandpass_filter_v2(sample, 1, 40, 100)
    # sample_label = torch.tensor(np.copy(self.label[idx]))-1#torch.tensor(1 if self.main[idx]['label'] == 'work' else 0)
    sample_label = torch.tensor(0) if self.label[idx] == 1 else torch.tensor(1)
    # sample_label = label_map[sample['label']]

    channels = [mitsar_chls.index('T4'), mitsar_chls.index('T6')]
    sample = sample[:, [HSE_chls.index('T7'), HSE_chls.index('T8')]]
    # sample_min, sample_max = sample.min(0), sample.max(0)
    # sample = (sample - sample_min) / (sample_max - sample_min)
    sample[:, 0] -= self.norm['mean'][HSE_chls.index('T7')]
    sample[:, 1] -= self.norm['mean'][HSE_chls.index('T8')]
    sample[:, 0] /= self.norm['std'][HSE_chls.index('T7')]
    sample[:, 1] /= self.norm['std'][HSE_chls.index('T8')]

    # sample = butter_bandpass_filter_v2(sample, 1, 40, 100)
    sample = torch.from_numpy(sample[:3000].astype(np.float32)).clone()
    return {'anchor': sample,
            'label': sample_label,
            'channels': torch.tensor(channels)}
 # --------------------------------------------from v7-----------------------------------------------
def init_chnls():
    HSE_chls = ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
        'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6',
        'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2']

    HSE_chls = [i.upper() for i in HSE_chls]

    mitsar_chls = ['Fp1', 'Fp2', 'FZ', 'FCz', 'Cz', 'Pz', 'O1', 'O2', 'F3', 'F4',
                   'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4', 'T5', 'T6', 'A1', 'A2']
    mitsar_chls = [i.upper() for i in mitsar_chls]

    raf_chls = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
           'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2']
    raf_chls = [i.upper() for i in raf_chls]

    # return HSE_chls, mitsar_chls, raf_chls
    return HSE_chls, mitsar_chls


HSE_chls, mitsar_chls  = init_chnls()
def load_data_v7():
    basepath ='/home/evgeniy/eeg_data/v100'
    basepath2 = '/home/data/HSE_math_all/processed/v5'
    # basepath ='/home/data/HSE_math_all/processed/v7'
    # basepath_1 ='/home/data/HSE_math_all/processed/v7'

    train_data = np.load('/home/data/HSE_math_all//processed/v7/train_signal.non_filtered.precise_split.256Hz.npy',allow_pickle=True)
    test_data = np.load('/home/data/HSE_math_all/processed/v7/test_signal.non_filtered.precise_split.256Hz.npy',allow_pickle=True)

    train_label = np.load('/home/data/HSE_math_all/processed/v7/train_label.REMAP.baselined_v2.non_filtered.precise_split.256Hz.npy',allow_pickle=True)
    test_label = np.load('/home/data/HSE_math_all/processed/v7/test_label.REMAP.baselined_v2.non_filtered.precise_split.256Hz.npy',allow_pickle=True)

    _train_label = np.load('/home/data/HSE_math_all/processed/v7/train_label.non_filtered.precise_split.256Hz.npy',allow_pickle=True)
    _test_label = np.load('/home/data/HSE_math_all/processed/v7/test_label.non_filtered.precise_split.256Hz.npy',allow_pickle=True)

    train_correct = np.load('/home/data/HSE_math_all/processed/v7/train_correct.non_filtered.precise_split.256Hz.npy',allow_pickle=True)
    test_correct = np.load('/home/data/HSE_math_all/processed/v7/test_correct.non_filtered.precise_split.256Hz.npy',allow_pickle=True)

    train_mean_std = np.load('/home/data/HSE_math_all/processed/v7/train_mean_std.non_filtered.precise_split.256Hz.npy',allow_pickle=True)
    test_mean_std = np.load('/home/data/HSE_math_all/processed/v7/test_mean_std.non_filtered.precise_split.256Hz.npy',allow_pickle=True)
    #





#     train_data = np.load(basepath + '/train_signal.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#     test_data = np.load(basepath + '/test_signal.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
# # - размеры данных и меток не совпадают!!!
#     train_label = np.load(basepath + '/train_label.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#     test_label = np.load(basepath + '/test_label.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#
#     _train_label = np.load(basepath + '/train_label.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#     _test_label = np.load(basepath + '/test_label.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#
#
#     train_correct = np.load(basepath + '/train_correct.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#     test_correct = np.load(basepath + '/test_correct.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#
#     train_mean_std = np.load(basepath + '/train_mean_std.non_filtered.precise_split.256Hz.npy', allow_pickle=True)
#     test_mean_std = np.load(basepath + '/test_mean_std.non_filtered.precise_split.256Hz.npy', allow_pickle=True)

    len(train_data), len(train_label)

    train_data = [val for i, val in enumerate(train_data) if _train_label[i] not in [0]]
    train_correct = [val for i, val in enumerate(train_correct) if _train_label[i] not in [0]]
    train_mean_std = [val for i, val in enumerate(train_mean_std) if _train_label[i] not in [0]]
    # train_label = [val for i, val in enumerate(train_label) if val not in [0]]

    test_data = [val for i, val in enumerate(test_data) if _test_label[i] not in [0]]
    test_correct = [val for i, val in enumerate(test_correct) if _test_label[i] not in [0]]
    test_mean_std = [val for i, val in enumerate(test_mean_std) if _test_label[i] not in [0]]
    # test_label = [val for i, val in enumerate(test_label) if val not in [0]]

    len(train_data), len(train_label)

    channels_meta = {'mean': [], 'std': []}
    channels_meta['mean'] = (np.concatenate([train_data, test_data]).reshape(np.concatenate([train_data, test_data]).shape[0] * np.concatenate([train_data, test_data]).shape[1], -1).mean(0))
    channels_meta['std'] = (np.concatenate([train_data, test_data]).reshape(np.concatenate([train_data, test_data]).shape[0] * np.concatenate([train_data, test_data]).shape[1], -1).std(0))

    train_data_ts = np.load(basepath2 + '/train_signal.non_filtered.precise_split.al.256Hz.npy', allow_pickle=True)
    test_data_ts = np.load(basepath2 + '/test_signal.non_filtered.precise_split.al.256Hz.npy', allow_pickle=True)

    train_label_ts = np.load(basepath2 + '/train_label.non_filtered.precise_split.al.256Hz.npy', allow_pickle=True)
    test_label_ts = np.load(basepath2 + '/test_label.non_filtered.precise_split.al.256Hz.npy', allow_pickle=True)

    train_mean_std_ts = np.load(basepath2 + '/train_mean_std.non_filtered.precise_split.al.256Hz.npy', allow_pickle=True)
    test_mean_std_ts = np.load(basepath2 + '/test_mean_std.non_filtered.precise_split.al.256Hz.npy', allow_pickle=True)


    channels_meta_ts = {'mean': [], 'std': []}
    channels_meta_ts['mean'] = (np.concatenate([train_data_ts.tolist() + test_data_ts.tolist()]).reshape(-1, 28).mean(0))
    channels_meta_ts['std'] = (np.concatenate([train_data_ts.tolist() + test_data_ts.tolist()]).reshape(-1, 28).std(0))

    return train_data, train_label, train_mean_std, train_correct, test_data, test_label, test_mean_std, test_correct, test_data_ts, test_label_ts, test_mean_std_ts, train_data_ts, train_label_ts, train_mean_std_ts


def dataset_upload():
  # dataset_root = '/home/data/HSE_exp'
  dataset_root = '/home/data/HSE_math_all'
  train_data = np.load(dataset_root + '/processed/v2/train_signal.npy', allow_pickle=True)
  test_data = np.load(dataset_root + '/processed/v2/test_signal.npy', allow_pickle=True)
  train_label = np.load(dataset_root + '/processed/v2/train_label.npy', allow_pickle=True)
  test_label = np.load(dataset_root + '/processed/v2/test_label.npy', allow_pickle=True)


  train_data = [val for i, val in enumerate(train_data) if train_label[i] not in [0, 2]]
  train_label = [val for i, val in enumerate(train_label) if val not in [0, 2]]

  test_data = [val for i, val in enumerate(test_data) if test_label[i] not in [0, 2]]
  test_label = [val for i, val in enumerate(test_label) if val not in [0, 2]]

  channels_meta = {'mean': [], 'std': []}
  channels_meta['mean'] = (np.concatenate([train_data, test_data]).reshape(np.concatenate([train_data, test_data]).shape[0] * np.concatenate([train_data, test_data]).shape[1], -1).mean(0))
  channels_meta['std'] = (np.concatenate([train_data, test_data]).reshape(np.concatenate([train_data, test_data]).shape[0] * np.concatenate([train_data, test_data]).shape[1], -1).std(0))

  # train_data3 = np.load(dataset_root + '/processed/v2/train_signal.3.npy', allow_pickle=True)
  # test_data3 = np.load(dataset_root + '/processed/v2/test_signal.3.npy', allow_pickle=True)
  #
  # train_label3 = np.load(dataset_root + '/processed/v2/train_label.3.npy', allow_pickle=True)
  # test_label3 = np.load(dataset_root + '/processed/v2/test_label.3.npy', allow_pickle=True)
  # train_data3 = [val for i, val in enumerate(train_data3) if train_label3[i] not in [0, 2]]
  # train_data3 = [val for i, val in enumerate(train_data3) if train_label3[i] not in [0, 2]]
  # train_label3 = [val for i, val in enumerate(train_label3) if val not in [0, 2]]
  #
  # test_data3 = [val for i, val in enumerate(test_data3) if test_label3[i] not in [0, 2]]
  # test_label3 = [val for i, val in enumerate(test_label3) if val not in [0, 2]]
  # channels_meta3 = {'mean': [], 'std': []}
  # channels_meta3['mean'] = (np.concatenate([train_data3, test_data3]).reshape(np.concatenate([train_data3, test_data3]).shape[0] * np.concatenate([train_data3, test_data3]).shape[1], -1).mean(0))
  # channels_meta3['std'] = (np.concatenate([train_data3, test_data3]).reshape(np.concatenate([train_data3, test_data3]).shape[0] * np.concatenate([train_data3, test_data3]).shape[1], -1).std(0))

  train_data, train_label, train_mean_std, train_correct, test_data, test_label, test_mean_std, test_correct, test_data_ts, test_label_ts, test_mean_std_ts, train_data_ts, train_label_ts, train_mean_std_ts = load_data_v7()

  print("train_data", len(train_data),"test_data", len(test_data))

  # train_dataset = TEST(train_data, train_label, channels_meta)
  train_dataset = TEST_v7(train_data, train_label, train_mean_std, HSE_chls, train_correct)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True,
                                             worker_init_fn=worker_init_fn)

  # test_dataset = TEST(test_data, test_label, channels_meta)
  test_dataset = TEST_v7(test_data, test_label, test_mean_std,HSE_chls, mitsar_chls )
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True, worker_init_fn = worker_init_fn)

  # test_dataset3 = TEST(test_data3, test_label3, channels_meta3)
  # test_loader3 = torch.utils.data.DataLoader(test_dataset3, batch_size=32, shuffle=False, num_workers=0, drop_last=True, worker_init_fn = worker_init_fn)

  return train_loader, test_loader #, test_loader3
#__________________________________________________________________________________________________________________________________________________
#                                         BERT MODEL
#__________________________________________________________________________________________________________________________________________________
from bert_conv_custom import BertConfig, BertEncoder
import math
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class TransposeCustom(torch.nn.Module):
    def __init__(self):
        super(TransposeCustom, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)

class EEGEmbedder(torch.nn.Module):
    def __init__(self):
        super(EEGEmbedder, self).__init__()
        config = BertConfig(is_decoder=False,
                            add_cross_attention=False,
                            ff_layer='linear',
                            hidden_size=768,
                            num_attention_heads=8,
                            num_hidden_layers=3,
                            conv_kernel=1,
                            conv_kernel_num=1)
        self.model = BertEncoder(config)

        self.pos_e = PositionalEncoding(512, max_len=6000)
        self.ch_embedder = torch.nn.Embedding(len(mitsar_chls), 512)
        self.ch_norm = torch.nn.LayerNorm(512)

        self.input_norm = torch.nn.LayerNorm(2)
        self.input_embedder = torch.nn.Sequential(
            TransposeCustom(),
            torch.nn.Conv1d(2, 32, 5, 2, padding=0),
            torch.nn.Conv1d(32, 64, 5, 2, padding=0),
            # TransposeCustom(),
            torch.nn.GroupNorm(64 // 2, 64),
            torch.nn.GELU(),
            # TransposeCustom(),
            torch.nn.Conv1d(64, 128, 3, 2, padding=0),
            torch.nn.Conv1d(128, 196, 3, 2, padding=0),
            # TransposeCustom(),
            torch.nn.GroupNorm(196 // 2, 196),
            torch.nn.GELU(),
            # TransposeCustom(),
            torch.nn.Conv1d(196, 256, 5, 1, padding=0),
            torch.nn.Conv1d(256, 384, 5, 1, padding=0),
            # TransposeCustom(),
            torch.nn.GroupNorm(384 // 2, 384),
            torch.nn.GELU(),
            # TransposeCustom(),
            torch.nn.Conv1d(384, 512, 5, 1, padding=0),
            torch.nn.Conv1d(512, 768, 1, 1, padding=0),
            torch.nn.GroupNorm(768 // 2, 768),
            torch.nn.GELU(),
            TransposeCustom(),
            # torch.nn.LeakyReLU(),
        )
        # self.input_norm = torch.nn.LayerNorm(10)
        # self.output_embedder = torch.nn.Conv1d(512, 512, 1)
        self.output_embedder = torch.nn.Linear(512, 512)
        self.transpose = TransposeCustom()

        self.mask_embedding = torch.nn.Parameter(torch.normal(0, 512 ** (-0.5), size=(512,)),
                                                 requires_grad=True)
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.Linear(256, 2),
            # torch.nn.Sigmoid()
        )

    def single_forward(self, inputs, attention_mask, ch_vector, placeholder):
        embedding = self.input_embedder(inputs)
        # create embedding for two channel indexes and sumup them to a single one
        # ch_embedding = self.ch_embedder(ch_vector).sum(1)
        # ch_embedding = ch_embedding[:, None]
        # print(embedding.shape, ch_embedding.shape)
        # embedding += ch_embedding
        # embedding = self.ch_norm(embedding)
        # we lost some channel specific information
        # embedding_unmasked = embedding.clone()

        # mask = _make_mask((embedding.shape[0], embedding.shape[1]), 0.05, embedding.shape[1], 10)
        # embedding[mask] = self.mask_embedding

        # for b_i in range(embedding.shape[0]):
        #     embedding_masked[b_i][mask[b_i]] = self.mask_embedding

        # embedding = torch.cat([placeholder, embedding], 1)
        encoder_output = self.model(embedding, output_hidden_states=True,
                                    output_attentions=True)[0]
        # encoder_output = self.output_embedder(encoder_output)
        # encoder_output = self.transpose(encoder_output)
        return torch.sum(encoder_output, 1), None

    def forward(self, a, mask, ch_vector, placeholder):
        a_downsampled_embedding, label = self.single_forward(a, mask, ch_vector, placeholder)
        pred = self.classification(a_downsampled_embedding)
        return pred, label

    def infer(self, a, ch_vector, placeholder):
        embedding = self.input_embedder(inputs)

        # create embedding for two channel indexes and sumup them to a single one
        ch_embedding = self.ch_embedder(ch_vector).sum(1)
        ch_embedding = ch_embedding[:, None]
        # print(embedding.shape, ch_embedding.shape)
        embedding += ch_embedding
        # embedding = self.ch_norm(embedding)

        embedding = torch.cat([placeholder, embedding], 1)
        encoder_output = self.model(embedding, output_hidden_states=True,
                                    output_attentions=True)[0]

        encoder_output = self.output_embedder(self.transpose(encoder_output))
        return encoder_output, embedding
#__________________________________________________________________________________________________________________________________________________
#                                         CLASSIFICATION MODEL
#__________________________________________________________________________________________________________________________________________________

import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        # self.wav2vec2.config_class.mask_time_length = 5
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# from typing import Any, Dict, Union
from typing import Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch


def train_bert_eeg_short():
    model = EEGEmbedder()
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # train_dataset = TEST(train_data, train_label, train_mean_std, HSE_chls, train_correct)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0,
    #                                            drop_last=True, worker_init_fn=worker_init_fn)
    #
    # test_dataset = TEST(test_data, test_label, test_mean_std, HSE_chls, test_correct)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True,
    #                                           worker_init_fn=worker_init_fn)

    train_loader, test_loader = dataset_upload()

    model.train()

    lr_d = 1e-6

    training_epochs1 = 10000 // len(train_loader)

    optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()


    for epoch in range(training_epochs1):
        train_loss_list1 = []
        for batch in train_loader:
            # batch = train_dataset.__getitem__(i)
            optim.zero_grad()
            placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            ae, _ = model(
                batch['anchor'].to(device),
                None,
                batch['channels'].long().to(device),
                placeholder.to(device))
            loss = loss_func(ae.view(-1, 2), batch['label'].to(device).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            mean_loss = loss.item()
            train_loss_list1.append(mean_loss)
            optim.step()

        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list1)))


def main1():  # тренируем сеть Тимура
    # model, test_loader = train_bert_eeg_short()
    # return

    train_loader, test_loader = dataset_upload()
    model = EEGEmbedder()
    device = 'cuda:0'
    loss_func = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    lr_d = 1e-6
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    num_of_epoch = 50

    for epoch in range(0,num_of_epoch):
        train_loss_list = []
        for batch in train_loader:
            optim.zero_grad()
            placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            ae, _ = model(
                batch['anchor'].to(device),
                None,
                batch['channels'].long().to(device),
                placeholder.to(device))
            logits = ae.view(-1, 2)
            # logits = model(batch['anchor'][:, :, 0].to('cuda:0')).logits
            results = torch.argmax(logits, dim=1).float()
            results.requires_grad = True
            loss = loss_func(logits,batch['label'].to('cuda:0').long())
            # loss = loss_func(results,batch['label'].to('cuda:0').float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # нормализация градиентов вместе - уточнить у Тимура профит
            mean_loss = loss.item()
            train_loss_list.append(mean_loss)

            optim.step()
        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list)))

from transformers import (
    Trainer,
    is_apex_available,
)

from typing import Any
class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


class CTCTrainer_EEG(Trainer):                                # !!!!!!!!!!!!!!!!!!!!!!!!! Вернуть при EEG !!!!!!!!!!!
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        # inputs = self._prepare_inputs(inputs)
        loss_func = torch.nn.CrossEntropyLoss()
        logits = model(batch['anchor'][:, :, 0].to('cuda:0')).logits
        loss = loss_func(logits, batch['label'].to('cuda:0').long())


        return loss.detach()

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

def Emo_audio_data_preparation():
    from tqdm import tqdm
    import os
    import torchaudio
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    data = []

    for path in tqdm(Path("/home/evgeniy/audio_datasets/aesdd").glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = str(path).split('/')[-2]

        try:
            # There are some broken files
            s = torchaudio.load(path)
            data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            # print(str(path), e)
            pass
    df = pd.DataFrame(data)
    df.head()

    # Filter broken and non-existed paths
    print(f"Step 0: {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")

    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    df.head()
    print("Labels: ", df["emotion"].unique())
    print()
    df.groupby("emotion").count()[["path"]]

    save_path = "/home/evgeniy/audio_datasets/aesdd_train_and_test"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["emotion"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

    print(train_df.shape)
    print(test_df.shape)


    return train_df, test_df

def Emo_audio_data_upload():
    save_path = "/home/evgeniy/audio_datasets/aesdd_train_and_test"
    data_files = {
        "train": save_path + "/train.csv",
        "validation": save_path + "/test.csv",
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(train_dataset)
    print(eval_dataset)

    return train_dataset, eval_dataset

def speech_file_to_array_fn(path):
    import torchaudio
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label
#------------------------------------------------out of main for EMO --------------------------------------------------------------------------
# We need to specify the input and output column
# we need to distinguish the unique labels in our SER dataset

train_dataset, eval_dataset = Emo_audio_data_upload()
input_column = "path"
output_column = "emotion"
label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes: {label_list}")
# model_name_or_path = "lighteternal/wav2vec2-large-xlsr-53-greek"
model_name_or_path = "/home/evgeniy/models/wav2vec2-large-xlsr-53-greek"
def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate, padding=True)
    result["labels"] = list(target_list)

    return result


from transformers import EvalPrediction
def compute_metrics_0(p: EvalPrediction):                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Вернуть для EEG!!!!!!!!!!
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def compute_metrics(p: EvalPrediction):
    is_regression = False
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path, )
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

def main():    # тренируем классификатор wav2vec на основе эмоций
    # model, test_loader = train_bert_eeg_short()
    # return
    # train_df, test_df = Emo_audio_data_preparation()
    # global processor, label_list, input_column, output_column, target_sampling_rate



    # train_loader, test_loader = dataset_upload()
    # train_data, train_label, train_mean_std, train_correct, test_data, test_label, test_mean_std, test_correct, test_data_ts, test_label_ts, test_mean_std_ts, train_data_ts, train_label_ts, train_mean_std_ts = load_data_v7()

    model_name_or_path = "/home/evgeniy/models/wav2vec2-large-xlsr-53-greek"

    # config
    from transformers import AutoConfig, Wav2Vec2Processor
    pooling_mode = "mean"
    # label_list = [0, 1, 2, 3]
    # label_list = [0, 1]                                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! это надо будет вернуть для EEG
    num_labels = len(label_list)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        # mask_time_length=2,                              # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! это надо будет вернуть для EEG
    )
    setattr(config, 'pooling_mode', pooling_mode)

    global train_dataset, eval_dataset

    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )

    model.freeze_feature_extractor()

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        # output_dir="/content/wav2vec2-xlsr-greek-speech-emotion-recognition",
        output_dir="/home/evgeniy/Output/wav2vec2-xlsr-greek-speech-emotion-recognition",
        # output_dir="/content/gdrive/MyDrive/wav2vec2-xlsr-greek-speech-emotion-recognition"
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=1.0,
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=2,
        report_to="none",                       # turn off wandb !!!!!!!!!!------------------
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_data,
    #     eval_dataset=test_data,
    #     compute_metrics=compute_metrics,
    # )

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    return

    device = 'cuda:0'
    loss_func = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    lr_d = 1e-6
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    num_of_epoch = 50

    for epoch in range(0,num_of_epoch):
        train_loss_list = []
        for batch in train_loader:
            optim.zero_grad()
            logits = model(batch['anchor'][:, :, 0].to('cuda:0')).logits
            loss = loss_func(logits,batch['label'].to('cuda:0').long())

            # results = torch.argmax(logits, dim=1).float()
            # results.requires_grad = True
            # loss = loss_func(logits,batch['label'].to('cuda:0').long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # нормализация градиентов вместе - уточнить у Тимура профит
            mean_loss = loss.item()
            train_loss_list.append(mean_loss)

            optim.step()
        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list)))


def main2():

    train_loader, test_loader = dataset_upload()

    # import matplotlib.pyplot as plt
    # correct = []
    # for batch in train_loader:
    #     correct.extend(batch['label'])
    #
    # correct = np.array([i.tolist() for i in correct])
    # plt.plot(correct)
    #
    # plt.plot(correct[:500].argmax(-1))
    # plt.show()
    #
    # return

    from transformers import AutoConfig, Wav2Vec2Processor

    # config
    # config = AutoConfig.from_pretrained(model_name_or_path)
    pooling_mode = "mean"
    # label_list = [0, 1, 2, 3]
    label_list = [0, 1]
    num_labels = len(label_list)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        mask_time_length=2,
    )
    setattr(config, 'pooling_mode', pooling_mode)


    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )

    # model = EEGEmbedder()

    device = 'cuda:0'

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('./logs')

    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    # optim = torch.optim.AdamW([{'params': model.wav2vec2.parameters(), 'lr': 1e-7},
    #                           {'params': model.classifier.parameters(), 'lr': 1e-4}])
    lr_d = 1e-4
    # optim = torch.optim.AdamW([{'params': model.wav2vec2.parameters(), 'lr': 1e-16},
    #                           {'params': model.classifier.parameters(), 'lr': 1e-2}])
    #
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    model.train()
    num_of_epoch = 50
    steps = 0
    # batchstep = 0
    train_loss = []
    val_loss = []
    val_loss_list = []
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model.freeze_feature_extractor()
    for epoch in range(0,num_of_epoch):

        # print("epoch:",epoch)
        train_loss_list = []
        for batch in train_loader:
            # if batchstep == 0:
            #     batch_fix = batch
            # batchstep +=1

            optim.zero_grad()
            # placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            # logits = model(batch['anchor'].to('cuda:0'), attention_mask=attention_mask).logits
            # batchsize = len(batch['anchor'])
            # batch_size = batch['anchor'].shape[0]
            # chanales_num = batch['anchor'].shape[2]
            # for b in [0, batchsize]:
              # for ch in [0, chanales_num]:
              # features = processor(batch['anchor'][b,:,0], sampling_rate=processor.feature_extractor.sampling_rate,
              #                      return_tensors="pt",
              #                      padding=True)
              # input_values = features.input_values.to(device)
              # attention_mask = features.attention_mask.to(device)

              # logits = model(batch['anchor'][b,:,0][None,:].to('cuda:0')).logits
            placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            # ae, _ = model(
            #     batch['anchor'].to(device),
            #     None,
            #     batch['channels'].long().to(device),
            #     placeholder.to(device))
            # logits = ae.view(-1, 2)
            logits = model(batch['anchor'][:, :, 0].to('cuda:0')).logits
            # results = torch.argmax(logits, dim=1).float()
            # results.requires_grad = True
            # loss = loss_func(results,batch['label'].to('cuda:0').float())
            loss = loss_func(logits,batch['label'].to('cuda:0').long())
            # loss = loss_func(torch.argmax(logits, dim=1),batch['label'].to('cuda:0'))
            # loss = loss_func(torch.argmax(logits, dim=1).float(),batch['label'].to('cuda:0').float())
            # loss = loss_func(torch.argmax(logits, dim=1).long(),batch['label'].to('cuda:0').long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # нормализация градиентов вместе - уточнить у Тимура профит
            mean_loss = loss.item()
            train_loss_list.append(mean_loss)

            optim.step()
            steps +=1
            # if steps % 10 == 0:
            #     print('epoch',epoch,'Loss: {}\t'.format(mean_loss))

            if steps != 0 and steps % 30000 == 0:
                der = 0
                preds = []
                reals = []
                val_loss_list = []
                try:
                    with torch.no_grad():
                        for batch in test_loader:
                            # batch = test_dataset.__getitem__(i)
                            # placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
                            logits = model(batch['anchor'][:, :, 0].to('cuda:0')).logits

                            reals.extend(batch['label'])
                            preds.extend(torch.argmax(logits, dim=1))

                            loss = loss_func(logits, batch['label'].to('cuda:0').long())
                            # loss = loss_func(torch.argmax(logits, dim=1), batch['label'].to('cuda:0').long())

                            loss = loss.mean()
                            val_loss_list.append(loss)
                            der += loss
                    der /= len(test_loader)
                    # writer.add_scalar('Loss/test', der, steps)


                    reals = np.array([i.tolist() for i in reals])
                    preds = np.array([i.tolist() for i in preds])

                    print(precision_recall_fscore_support(reals, preds))

                    print('Loss: {}\t'.format(der),'steps:',steps)

                except:
                    raise

        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list)))
        # print('list',train_loss_list)
        train_loss.append(np.mean(train_loss_list))
        val_loss.append(np.mean(val_loss_list))


#-----------------------------show train process ---------------------------
    import matplotlib.pyplot as plt
    n_epoch = len(train_loss)
    epochs = range(1, n_epoch + 1)

    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)

    xticks = range(1, n_epoch + 1, n_epoch // 10) if n_epoch > 10 else epochs

    # draw loss
    ax_top.plot(epochs, train_loss, 'r', label='train')
    ax_top.plot(epochs, val_loss, 'b', label='validation')

    ax_top.set(
        title='Loss',
        xlabel='Epoch number',
        ylabel='Loss value',
        ylim=[0, max(max(train_loss), max(val_loss)) + 1],
    )
    ax_top.legend(
        title="Выборка",
    )
    ax_top.grid()

    # # draw accuracy
    # ax_bottom.plot(epochs, train_fscore, 'r', label='train')
    # ax_bottom.plot(epochs, val_fscore, 'b', label='validation')
    #
    # ax_bottom.set(
    #     title='Accuracy',
    #     xlabel='Epoch number',
    #     ylabel='Accuracy value',
    #     xticks=xticks,
    #     ylim=[0, 1],
    # )
    # ax_bottom.legend(
    #     title="Выборка",
    # )
    # ax_bottom.grid()

    fig.suptitle("Кривые обучения")

    plt.show()



if __name__ == '__main__':
    main()