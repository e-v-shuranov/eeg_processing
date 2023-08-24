# based on:
# /home/sokhin/notebooks/ClassifiersSimple/Workload/actual/v5/baselined.v2/EEG.post_unfiltered.precise_split.ts_train.256Hz.v7.modified_model.binary_over_distance.ipynb



# %config Completer.use_jedi = False
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ['http_proxy'] = "http://127.0.0.1:3128"
# os.environ['https_proxy'] = "http://127.0.0.1:3128"
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter

import torch
from bert_conv_custom import BertConfig, BertEncoder
from transformers import BertModel
import math
from scipy import signal

from scipy.signal import resample
from scipy.signal import butter, lfilter
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.signal import freqz
from torch.optim.lr_scheduler import _LRScheduler
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, balanced_accuracy_score

device = device = "cuda:1" if torch.cuda.is_available() else "cpu"
#----------------Architecture-------------------------------------------
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

def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)

def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


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

# ----------------------------------------Data-----------------------------------------------------
def _generate_negatives(z):
    """Generate negative samples to compare each sequence location against"""
    num_negatives = 20
    batch_size, feat, full_len = z.shape
    z_k = z.permute([0, 2, 1]).reshape(-1, feat)
    with torch.no_grad():
        # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
        negative_inds = torch.randint(0, full_len-1, size=(batch_size, full_len * num_negatives))
        # From wav2vec 2.0 implementation, I don't understand
        # negative_inds[negative_inds >= candidates] += 1

        for i in range(1, batch_size):
            negative_inds[i] += i * full_len

    z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, num_negatives, feat)
    return z_k, negative_inds


def masking(ts):
    start_shift = np.random.choice(range(10))
    downsampling = 2
    indices = np.random.choice(np.array(list(range(110)))[start_shift::10][::downsampling], 5, replace=False)
    masked_idx = []
    for i in indices:
        masked_idx.extend(range(i, i + 10))

    masked_idx = np.array(masked_idx)

    # mask = np.ones((6000, 2))
    # # desync some masked channels
    # ts_masked = ts.copy()
    # if np.random.choice([0, 1], p=[0.7, 0.3]):
    #     ts_masked[masked_idx, np.random.choice([0, 1])] *= 0
    # else:
    #     ts_masked[masked_idx] *= 0

    return None, masked_idx


def init_dicts():
    # ({0: tensor(2.4250), 1: tensor(1.9767)},
    #  {0: tensor(66.8642), 1: tensor(46.8854)})

    mean_dict = {0: (0.9631),
     1: (1.0248),
     2: (1.3041),
     3: (0.),
     4: (1.5822),
     5: (1.7250),
     6: (0.9935),
     7: (0.9548),
     8: (0.7488),
     9: (1.3948),
     10: (0.8879),
     11: (1.0527),
     12: (1.3401),
     13: (1.5541),
     14: (1.2600),
     15: (1.0487),
     16: (0.7529),
     17: (1.6566),
     18: (0.9272),
     19: (1.2238),
     20: (1.2619),
     21: (1.5236)}

    std_dict = {0: (64.1294),
     1: (64.1984),
     2: (45.9215),
     3: (0.),
     4: (45.1312),
     5: (51.7621),
     6: (43.5150),
     7: (39.7182),
     8: (46.8787),
     9: (49.0797),
     10: (52.2342),
     11: (51.9236),
     12: (50.7353),
     13: (52.1277),
     14: (48.8627),
     15: (42.7040),
     16: (46.5815),
     17: (60.2403),
     18: (41.6082),
     19: (44.6035),
     20: (82.8107),
     21: (53.5717)}

    writer = SummaryWriter('./logs')
    return  mean_dict, std_dict, writer

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_v2(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter_v2(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass_v2(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


class TEST(torch.utils.data.Dataset):
    def __init__(self, main, labels, norm, chls, correct):
        super(TEST, self).__init__()
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

def load_data():
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



# ---------------------------------Training-----------------------------------------------------------------------
train_data, train_label, train_mean_std, train_correct, test_data, test_label, test_mean_std, test_correct, test_data_ts, test_label_ts, test_mean_std_ts, train_data_ts, train_label_ts, train_mean_std_ts = load_data()

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    np.random.seed((torch_seed + worker_id) % 2 ** 30)

def train_bert_eeg_short():
    model = EEGEmbedder()

    train_dataset = TEST(train_data, train_label, train_mean_std, HSE_chls, train_correct)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0,
                                               drop_last=True, worker_init_fn=worker_init_fn)

    test_dataset = TEST(test_data, test_label, test_mean_std, HSE_chls, test_correct)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True,
                                              worker_init_fn=worker_init_fn)
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



def train_bert_eeg():
    model = EEGEmbedder()
    # mitsar_chls, HSE_chls = init_chnls()
    len(train_data)


    class NoamLR(_LRScheduler):
        def __init__(self, optimizer, warmup_steps, d_model=512):
            self.warmup_steps = warmup_steps
            self.d_model = d_model
            super().__init__(optimizer)

        def get_lr(self):
            last_epoch = max(1, self.last_epoch)
            factor = min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
            # scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
            # return [base_lr * scale for base_lr in self.base_lrs]
            return [base_lr * self.d_model ** (-0.5) * factor for base_lr in self.base_lrs]


    cossim = torch.nn.CosineSimilarity(dim=-1)

    def cosloss(anchor, real, negative):
        a = torch.exp(cossim(anchor, real)) / 0.1
        b = sum([torch.exp(cossim(anchor, negative[:, n])) / 0.1 for n in range(negative.shape[1])]) + 1e-6
        return -torch.log(a / b)




    train_dataset = TEST(train_data, train_label, train_mean_std, HSE_chls, train_correct)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0,
                                               drop_last=True, worker_init_fn=worker_init_fn)

    test_dataset = TEST(test_data, test_label, test_mean_std, HSE_chls, test_correct)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True,
                                              worker_init_fn=worker_init_fn)

    train_dataset_ts = TEST(test_data_ts, test_label_ts, test_mean_std_ts, HSE_chls, None)
    train_loader_ts = torch.utils.data.DataLoader(train_dataset_ts, batch_size=32, shuffle=False, num_workers=0,
                                                  drop_last=True, worker_init_fn=worker_init_fn)

    # test_dataset_raf = TEST(test_data_raf, test_label_raf, channels_meta_raf, HSE_chls, test_correct_raf)
    # test_loader_raf = torch.utils.data.DataLoader(test_dataset_raf, batch_size=32, shuffle=False, num_workers=0, drop_last=True, worker_init_fn = worker_init_fn)

    # test_dataset_v4 = TEST(test_data_v4 , test_label_v4, channels_meta, HSE_chls, None)
    # test_loader_v4 = torch.utils.data.DataLoader(test_dataset_v4, batch_size=32, shuffle=False, num_workers=0, drop_last=True, worker_init_fn = worker_init_fn)

    model.train()

    train_loss = []
    val_loss = []

    train_fscore = []
    val_fscore = []


    lr_d = 1e-6
    acc_size = 1
    # training_epochs1 = 35000 // len(train_loader)
    training_epochs1 = 10000 // len(train_loader)

    # model_test = torch.nn.DataParallel(model)
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    # scheduler = NoamLR(optim, 5000, 512)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr_d, total_steps=training_epochs1*len(train_loader))
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    steps = 0

    len(train_loader), training_epochs1, training_epochs1 * len(train_loader)

    for epoch in range(training_epochs1):
        mean_loss = 0
        acc_step = 0
        train_loss_list = []
        preds_train = []
        reals_train = []
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
            # loss = loss.mean() / acc_size
            train_loss_list.append(loss.item())
            reals_train.extend((batch['label']))
            preds_train.extend((ae))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            mean_loss = loss.item()
            train_loss_list1.append(mean_loss)
            acc_step += 1
            steps += 1
            optim.step()
            # scheduler.step()
            # if steps % 500 == 0:
            #     print('Loss: {}\t'.format(mean_loss))

            if steps != 0 and steps % 1000 == 0:
                der = 0
                preds = []
                reals = []
                correct = []
                val_loss_list = []
                with torch.no_grad():
                    for batch in test_loader:
                        # batch = test_dataset.__getitem__(i)
                        placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
                        ae, label = model(
                            batch['anchor'].to(device),
                            None,
                            batch['channels'].long().to(device),
                            placeholder.to(device))
                        # loss_positive = loss_fct(ae, pe)
                        # loss_negative = loss_fct(ae, ne)
                        reals.extend((batch['label']))
                        preds.extend((ae))
                        correct.extend(batch['correct'])
                        loss = loss_func(ae.view(-1, 2), batch['label'].to(device).long())
                        val_loss_list.append(loss.item())

                        loss = loss.mean() / acc_size
                        der += loss
                der /= len(test_loader)
                # writer.add_scalar('Loss/test', der, steps)

                reals = np.array([i.tolist() for i in reals])
                preds = np.array([i.tolist() for i in preds])
                #             preds_ = np.copy(preds[:, 1])
                #             th = 0.5

                #             preds_[np.where(preds[:, 1] < th)] = 0
                #             preds_[np.where(preds[:, 1] >= th)] = 1
                fscore = precision_recall_fscore_support(reals, preds.argmax(-1))[2][1]
                print(precision_recall_fscore_support(reals, preds.argmax(-1)))
                # print('1 epoch:',epoch,'training_epochs1',training_epochs1)

                set(reals)
                # set(reals)
                val_loss.append(np.mean(val_loss_list))
                val_fscore.append(fscore)

                preds_train = np.array([i.tolist() for i in preds_train])
                reals_train = np.array([i.tolist() for i in reals_train])
                fscore = precision_recall_fscore_support(reals_train, preds_train.argmax(-1))[2][1]
                train_loss.append(np.mean(train_loss_list))
                train_fscore.append(fscore)

                train_loss_list = []
                preds_train = []
                reals_train = []

        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list1)))


    n_epoch = training_epochs1
# -------------- train additional training_epochs1 = 15000--------why?-----------------------------------------------------
#     model.train()
#
#     lr_d = 1e-6
#     acc_size = 1
#     # training_epochs1 = 15000 // len(train_loader)
#     training_epochs1 = 15000 // len(train_loader)
#     n_epoch = n_epoch + training_epochs1
#     # model_test = torch.nn.DataParallel(model)
#     optim = torch.optim.AdamW(model.parameters(), lr=lr_d)
#     # scheduler = NoamLR(optim, 5000, 512)
#     # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr_d, total_steps=training_epochs1*len(train_loader))
#     model.to(device)
#
#     loss_func = torch.nn.CrossEntropyLoss()
#
#     steps = 0
#
#     len(train_loader), training_epochs1, training_epochs1 * len(train_loader)
#
#
#     # model.cpu()(batch['anchor'][None], batch['mask'][None], batch['channels'][None])
#     for epoch in range(training_epochs1):
#         mean_loss = 0
#         acc_step = 0
#         for batch in train_loader:
#             # batch = train_dataset.__getitem__(i)
#             optim.zero_grad()
#             placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
#             ae, _ = model(
#                 batch['anchor'].to(device),
#                 None,
#                 batch['channels'].long().to(device),
#                 placeholder.to(device))
#             loss = loss_func(ae.view(-1, 2), batch['label'].to(device).long())
#             # loss = loss.mean() / acc_size
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             mean_loss = loss.item()
#             acc_step += 1
#             steps += 1
#             optim.step()
#             # scheduler.step()
#             if steps % 500 == 0:
#                 print('Loss: {}\t'.format(mean_loss))
#
#             if steps != 0 and steps % 1000 == 0:
#                 der = 0
#                 preds = []
#                 reals = []
#                 correct = []
#                 with torch.no_grad():
#                     for batch in test_loader:
#                         # batch = test_dataset.__getitem__(i)
#                         placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
#                         ae, label = model(
#                             batch['anchor'].to(device),
#                             None,
#                             batch['channels'].long().to(device),
#                             placeholder.to(device))
#                         # loss_positive = loss_fct(ae, pe)
#                         # loss_negative = loss_fct(ae, ne)
#                         reals.extend((batch['label']))
#                         preds.extend((ae))
#                         correct.extend(batch['correct'])
#                         loss = loss_func(ae.view(-1, 2), batch['label'].to(device).long())
#
#                         loss = loss.mean() / acc_size
#                         der += loss
#                 der /= len(test_loader)
#                 # writer.add_scalar('Loss/test', der, steps)
#
#                 reals = np.array([i.tolist() for i in reals])
#                 preds = np.array([i.tolist() for i in preds])
#                 #             preds_ = np.copy(preds[:, 1])
#                 #             th = 0.5
#
#                 #             preds_[np.where(preds[:, 1] < th)] = 0
#                 #             preds_[np.where(preds[:, 1] >= th)] = 1
#                 print(precision_recall_fscore_support(reals, preds.argmax(-1)))
#                 print('2 epoch:',epoch,'training_epochs1',training_epochs1)
#
#
#                 balanced_accuracy_score(reals, preds.argmax(-1))


#-----------------------------show train process ---------------------------

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

    # draw accuracy
    ax_bottom.plot(epochs, train_fscore, 'r', label='train')
    ax_bottom.plot(epochs, val_fscore, 'b', label='validation')

    ax_bottom.set(
        title='Accuracy',
        xlabel='Epoch number',
        ylabel='Accuracy value',
        xticks=xticks,
        ylim=[0, 1],
    )
    ax_bottom.legend(
        title="Выборка",
    )
    ax_bottom.grid()

    fig.suptitle("Кривые обучения")

    plt.show()
    # %%






    return model, test_loader

def check_results(model, test_loader):
    preds = []
    reals = []
    correct = []
    model = model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # batch = test_dataset.__getitem__(i)
            placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            ae, label = model(
                batch['anchor'].to(device),
                None,
                batch['channels'].long().to(device),
                placeholder.to(device))
            # loss_positive = loss_fct(ae, pe)
            # loss_negative = loss_fct(ae, ne)
            reals.extend(batch['label'])
            correct.extend(batch['correct'])
            preds.extend(ae.view(-1, 2))
            # loss = loss_func(ae.view(-1), batch['label'].to(device).float())

            # loss = loss.mean() / acc_size
            # der += loss
    # der /= len(test_loader)
    # writer.add_scalar('Loss/test', der, steps)

    reals = np.array([i.tolist() for i in reals])
    preds = np.array([i.tolist() for i in preds])

    print(precision_recall_fscore_support(reals, preds.argmax(-1)))  # расчет метрики!!!

    print('2 classes: precision, recall, fscore ,amount:',precision_recall_fscore_support(((reals > 0) * 1) * ((reals > 1) * 1 + 1), (preds.argmax(-1) + 1)))
    plt.plot(preds.argmax(-1))

    plt.rcParams["figure.figsize"] = (20, 6)

    plt.plot(preds.argmax(-1)[:200], label='Predict')
    plt.plot(reals[:200] + 2.1, label='Remapped labels')
    plt.xlabel("Time steps (3 sec each frame)")
    plt.yticks([0, 1, 2])
    plt.legend()

    # plt.plot(preds[1000:1500, 2] )             #IndexError: index 2 is out of bounds for axis 1 with size 2
    # plt.plot(reals[1000:1500] * 50)
    plt.show()


    len(train_data_ts), len(test_data_ts)

    train_dataset_ts = TEST(train_data_ts, train_label_ts, train_mean_std_ts, HSE_chls, None)
    train_loader_ts = torch.utils.data.DataLoader(train_dataset_ts, batch_size=32, shuffle=False, num_workers=0,
                                                  drop_last=True, worker_init_fn=worker_init_fn)

    preds = []
    reals = []
    correct = []
    model = model.train()
    with torch.no_grad():
        for batch in train_loader_ts:
            # batch = test_dataset.__getitem__(i)
            placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            ae, label = model(
                batch['anchor'].to(device),
                None,
                batch['channels'].long().to(device),
                placeholder.to(device))
            # loss_positive = loss_fct(ae, pe)
            # loss_negative = loss_fct(ae, ne)
            reals.extend(batch['pure_label'])
            correct.extend(batch['correct'])
            preds.extend(ae.view(-1, 2))                       # evgeniy replaced             preds.extend(ae.view(-1, 3))
            # preds.extend(ae.view(-1, 3))
            # loss = loss_func(ae.view(-1), batch['label'].to(device).float())

            # loss = loss.mean() / acc_size
            # der += loss
    # der /= len(test_loader)
    # writer.add_scalar('Loss/test', der, steps)

    reals = np.array([i.tolist() for i in reals])
    preds = np.array([i.tolist() for i in preds])
    correct = np.array([i.tolist() for i in correct])
    # preds[np.where(preds < 2.5)] = -1
    # preds[np.where(preds >= 2.5)] = 1
    # print(precision_recall_fscore_support(reals, preds))

    plt.plot(preds)

    preds.shape

    plt.plot(preds[:500].argmax(-1))
    plt.plot(reals[:500])

    plt.plot(preds[:500] / 1)
    plt.plot(reals[:500])
    plt.show()
def main():
    model, test_loader = train_bert_eeg_short()

    torch.save(model, '/home/evgeniy/eeg_processing/models/model_v2.npy')
    #
    test_dataset = TEST(test_data, test_label, test_mean_std, HSE_chls, test_correct)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True,
                                              worker_init_fn=worker_init_fn)
    # model = torch.load('/home/evgeniy/eeg_processing/models/model_v1.npy')
    model.eval()

    check_results(model, test_loader)



if __name__ == '__main__':
    main()