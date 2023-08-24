import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor

from dataclasses import dataclass

import os

from typing import Tuple

from typing import Dict, List, Optional, Union

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from transformers.file_utils import ModelOutput

import torch.nn as nn
from packaging import version


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True

from transformers import EvalPrediction
import torchaudio


import numpy as np


mitsar_chls = ['Fp1', 'Fp2', 'FZ', 'FCz', 'Cz', 'Pz', 'O1', 'O2', 'F3', 'F4',
               'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4', 'T5', 'T6', 'A1', 'A2']
mitsar_chls = [i.upper() for i in mitsar_chls]

HSE_chls = ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6',
    'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2']

HSE_chls = [i.upper() for i in HSE_chls]

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



@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


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
                loss_fct = torch.nn.CrossEntropyLoss()             #CrossEntropyLoss()
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
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def compute_metrics(p: EvalPrediction):
    is_regression = False
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


import random
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    np.random.seed((torch_seed + worker_id) % 2**30)

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
# config = BertConfig(is_decoder=True,
#                     add_cross_attention=True,
#                     ff_layer='conv',
#                     conv_kernel=1,
#                     conv_kernel_num=3)

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_bert_eeg_short_bert():
    model_bert = EEGEmbedder()

    train_dataset = TEST(train_data, train_label, train_mean_std, HSE_chls, train_correct)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0,
                                               drop_last=True, worker_init_fn=worker_init_fn)

    # test_dataset = TEST(test_data, test_label, test_mean_std, HSE_chls, test_correct)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True,
    #                                           worker_init_fn=worker_init_fn)
    model_bert.train()

    lr_d = 1e-6

    # training_epochs1 = 10000 // len(train_loader)

    optim_bert = torch.optim.AdamW(model_bert.parameters(), lr=lr_d, weight_decay=1)
    model_bert.to(device)

    loss_func_bert = torch.nn.CrossEntropyLoss()


    num_of_epoch = 50
    for epoch in range(0,num_of_epoch):
        train_loss_list_bert = []
        for batch in train_loader:
            # batch = train_dataset.__getitem__(i)
            optim_bert.zero_grad()
            placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            ae, _ = model_bert(
                batch['anchor'].to(device),
                None,
                batch['channels'].long().to(device),
                placeholder.to(device))
            loss_bert = loss_func_bert(ae.view(-1, 2), batch['label'].to(device).long())
            loss_bert.backward()
            torch.nn.utils.clip_grad_norm_(model_bert.parameters(), 1.0)
            mean_loss_bert = loss_bert.item()
            train_loss_list_bert.append(mean_loss_bert)
            optim_bert.step()

        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list_bert)))

def TUH_dataset_upload():
    splitted_paths = ['/home/data/TUH_pretrain.filtered_1_40.v2.splited/{}'.format(i) for i in os.listdir('/home/data/TUH_pretrain.filtered_1_40.v2.splited/')]
    print(len(splitted_paths))
    test_dataset = TEST_v7(splitted_paths)
    train_dataset = TEST_v7(train_data, train_label, train_mean_std, HSE_chls, train_correct)

def main():
    # model_bert, test_loader_bert = train_bert_eeg_short_bert()
    # return
    # TUH_dataset_upload()

    train_loader, test_loader = dataset_upload()

    # train_dataset, eval_dataset = Emo_audio_data_upload()
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
    #                                            drop_last=True,
    #                                            worker_init_fn=worker_init_fn)
    # test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=True,
    #                                           worker_init_fn=worker_init_fn)
# -----------------------------------------------------------------------------------wave2vec part--------------------
    loss_func = torch.nn.CrossEntropyLoss()
    model_name_or_path = "/home/evgeniy/models/wav2vec2-large-xlsr-53-greek"

    from transformers import AutoConfig, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    pooling_mode = "mean"
    output_column = "emotion"
    # label_list = train_dataset.unique(output_column)

    # label_list = [0, 1, 2, 3]
    label_list = [0, 1]                                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! это надо будет вернуть для EEG
    num_labels = len(label_list)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        mask_time_length=2,                              # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! это надо будет вернуть для EEG
    )
    setattr(config, 'pooling_mode', pooling_mode)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    ).to(device)

    # model = Wav2Vec2ForSpeechClassification(config=config).to(device)
    # model = Wav2Vec2ClassificationHead(config).to(device)
    model.train()
    lr_d = 1e-4
    optim = torch.optim.AdamW(model.classifier.parameters(), lr=lr_d, weight_decay=1)



    def speech_file_to_array_fn2(batch):
        speech_array, sampling_rate = torchaudio.load(batch)
        speech_array = torchaudio.transforms.Resample(sampling_rate, processor.feature_extractor.sampling_rate)((speech_array)).squeeze().numpy()
        return speech_array

    def get_emo_number(emo):
        return config.label2id[emo]
#
#     # attention_mask  - используем например в случае паддинга, ставим нули там где падим нулями. параметр не обязательный или задаем еденичками
# -----------------------------------------------------------------------------------bert part--------------------

    lr_d = 1e-6
    model_bert = EEGEmbedder()
    optim_bert = torch.optim.AdamW(model_bert.parameters(), lr=lr_d, weight_decay=1)
    model_bert.train()
    model_bert.to(device)
    loss_func_bert = torch.nn.CrossEntropyLoss()

    num_of_epoch = 50
    for epoch in range(0,num_of_epoch):
        train_loss_list = []
        train_loss_list_bert = []
        for batch in train_loader:
            optim.zero_grad()
            logits = model(batch['anchor'][:,:,0].to(device)).logits
            loss = loss_func(logits,batch['label'].to('cuda:0').long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # нормализация градиентов вместе - уточнить у Тимура профит
            mean_loss = loss.item()
            train_loss_list.append(mean_loss)
            optim.step()

            # optim_bert.zero_grad()
            # placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            # ae, _ = model_bert(
            #     batch['anchor'].to(device),
            #     None,
            #     batch['channels'].long().to(device),
            #     placeholder.to(device))
            # loss_bert = loss_func_bert(ae.view(-1, 2), batch['label'].to(device).long())
            # loss_bert.backward()
            # torch.nn.utils.clip_grad_norm_(model_bert.parameters(), 1.0)
            # mean_loss_bert = loss_bert.item()
            # train_loss_list_bert.append(mean_loss_bert)
            # optim_bert.step()

        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list)))
        # print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list_bert)))
        # print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list)), 'Loss_bert: {}\t'.format(np.mean(train_loss_list_bert)))

if __name__ == '__main__':
    main()