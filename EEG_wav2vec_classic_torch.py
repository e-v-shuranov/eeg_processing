import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor

from dataclasses import dataclass


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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = dataset_upload()

    # train_dataset, eval_dataset = Emo_audio_data_upload()
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
    #                                            drop_last=True,
    #                                            worker_init_fn=worker_init_fn)
    # test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=True,
    #                                           worker_init_fn=worker_init_fn)

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


    model.train()
    lr_d = 1e-6
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    num_of_epoch = 50


    def speech_file_to_array_fn2(batch):
        speech_array, sampling_rate = torchaudio.load(batch)
        speech_array = torchaudio.transforms.Resample(sampling_rate, processor.feature_extractor.sampling_rate)((speech_array)).squeeze().numpy()
        return speech_array

    def get_emo_number(emo):
        return config.label2id[emo]

    # attention_mask  - используем например в случае паддинга, ставим нули там где падим нулями. параметр не обязательный или задаем еденичками
    for epoch in range(0,num_of_epoch):
        train_loss_list = []
        for batch in train_loader:
            optim.zero_grad()
            # speech_array = list(map(speech_file_to_array_fn2, batch["path"]))
            # speech_array = batch['anchor'][:,:,0].detach().numpy()
            # features = processor(speech_array, sampling_rate=processor.feature_extractor.sampling_rate,
            #                      return_tensors="pt", padding=True)
            # input_values = features.input_values.to(device)
            # attention_mask = features.attention_mask.to(device)

            # output_values = list(map(get_emo_number, batch['emotion']))

            # logits = model(input_values, attention_mask=attention_mask).logits
            logits = model(batch['anchor'][:,:,0].to(device)).logits
            loss = loss_func(logits,batch['label'].to('cuda:0').long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # нормализация градиентов вместе - уточнить у Тимура профит
            mean_loss = loss.item()
            train_loss_list.append(mean_loss)

            optim.step()
        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list)))



if __name__ == '__main__':
    main()