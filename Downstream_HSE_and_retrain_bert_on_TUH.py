# origenaly based on:
# /home/sokhin/notebooks/ClassifiersSimple/Workload/actual/v5/baselined.v2/EEG.post_unfiltered.precise_split.ts_train.256Hz.v7.modified_model.binary_over_distance.ipynb
# updated on home/sokhin/notebooks/HSE_stage2/pretraining/Pretraining.v5.ipynb   25/08/2023
# updated on Pretraining.v6.ipynb  (Pretraining.v6_TUH_pretrain_final_14_09_2023.ipynb )



import time
# %config Completer.use_jedi = False
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# os.environ['http_proxy'] = "http://127.0.0.1:3128"
# os.environ['https_proxy'] = "http://127.0.0.1:3128"
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

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
from torch.utils.tensorboard import SummaryWriter

device = device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device = "cpu"
#----------------Architecture-------------------------------------------

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

    m_channels = ['Fp1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', \
                  'P3', 'Pz', 'P4', 'O1', 'O2']
    m_channels = [i.upper() for i in m_channels]
    TUH_chanels = ['FP1', 'FP2', 'FZ', 'FCZ', 'CZ', 'PZ', 'O1', 'O2', 'F3', 'F4', \
                   'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4', 'T5', 'T6', \
                   'A1', 'A2', 'EKG1']
    TUH_chanels = [i.upper() for i in TUH_chanels]
    #  Remove 'EKG1' and 'FCZ' based on pretrain dataloader TEST(): Pretraining.v5.ipynb
    TUH_chanels_for_training = ['FP1', 'FP2', 'FZ', 'CZ', 'PZ', 'O1', 'O2', 'F3', 'F4', \
                                'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4', 'T5', 'T6', \
                                'A1', 'A2']
    TUH_chanels_for_training = [i.upper() for i in TUH_chanels_for_training]

    #  TUH_chanels_for_training with  'FCZ' because model archetecture needs it in Pretraining.v5.ipynb
    TUH_chanels_for_training_plus3 = ['FP1', 'FP2', 'FZ', 'FCz', 'CZ', 'PZ', 'O1', 'O2', 'F3', 'F4', \
                                      'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4', 'T5', 'T6', \
                                      'A1', 'A2']
    TUH_chanels_for_training_plus3 = [i.upper() for i in TUH_chanels_for_training_plus3]

    HSE_Stage2_channels = ['Fp1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', \
                           'P3', 'Pz', 'P4', 'O1', 'O2']
    HSE_Stage2_channels = [i.upper() for i in HSE_Stage2_channels]

    # return HSE_chls, mitsar_chls, raf_chls
    return HSE_chls, mitsar_chls, HSE_Stage2_channels, TUH_chanels, TUH_chanels_for_training, TUH_chanels_for_training_plus3


HSE_chls, mitsar_chls, HSE_Stage2_channels, TUH_chanels, TUH_chanels_for_training, TUH_chanels_for_training_plus3  = init_chnls()




class TransposeCustom(torch.nn.Module):
    def __init__(self):
        super(TransposeCustom, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
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


class InputEmbedder(torch.nn.Module):
    def __init__(self, hidden_size=128, chnls=len(mitsar_chls)):
        super().__init__()

        self.hidden_size = hidden_size
        self.chnls = chnls
        self.time_reduced = 0
        #  ~128 points for spectr
        #  almost 256 receptive field, because it takes ~8.3% from 3000 points that ~ 249
        # input [batch x time x ch ]  output [batch x time_reduced x hidden_size x ch]    time reduction ~38 times   for  3000 it should be 80
        self.embedder = torch.nn.Sequential( # проверить по какому измерению идет свертка!!
            TransposeCustom(),                                                                                                         # (7/3000 + 7/1500 + 7/750 + 7/375 + 9/187)*3000 = 255 receptive field
            torch.nn.Conv1d(in_channels=chnls, out_channels=hidden_size//16 * chnls, groups=chnls, kernel_size=7, dilation=1, stride=2),   # 7 from 3000 receptive field
            torch.nn.Conv1d(in_channels=hidden_size //16 * chnls, out_channels=hidden_size //8 * chnls, groups=hidden_size//16 * chnls, kernel_size=7, dilation=1, stride=2),   # 7 from 1500 receptive field
            torch.nn.Conv1d(in_channels=hidden_size //8 * chnls, out_channels=hidden_size //4 * chnls, groups=hidden_size //8 * chnls, kernel_size=7, dilation=1, stride=2),  # 7 from 750 receptive field
            torch.nn.Conv1d(in_channels=hidden_size // 4 * chnls, out_channels=hidden_size // 2 * chnls, groups=hidden_size // 4 * chnls, kernel_size=7, dilation=1, stride=2), # 7 from 375 receptive field
            torch.nn.Conv1d(in_channels=hidden_size // 2 * chnls, out_channels=hidden_size * chnls, groups=hidden_size // 2 * chnls, kernel_size=9, dilation=1, stride=2), # 9 from 187 receptive field
        )

        # self.embedder = torch.nn.Sequential(  # Вариант с 345 отсчетами на выходе 249 receptive field.   Надо сохранить респтив филд но уменьшить количесвто отсчетов до ~80
        #     TransposeCustom(),  # (7/3000 + 19/1500 + 51/750)*3000 = 249 receptive field
        #     torch.nn.Conv1d(in_channels=chnls, out_channels=hidden_size // 16 * chnls, groups=chnls, kernel_size=7, dilation=1, stride=2),  # 7 from 3000 receptive field
        #     torch.nn.Conv1d(in_channels=hidden_size // 16 * chnls, out_channels=hidden_size // 4 * chnls, groups=hidden_size // 16 * chnls, kernel_size=7, dilation=3, stride=2),  # 19 from 1500 receptive field
        #     torch.nn.Conv1d(in_channels=hidden_size // 4 * chnls, out_channels=hidden_size * chnls, groups=hidden_size // 4 * chnls, kernel_size=11, dilation=5, stride=2),  # 51 from 750 receptive field
        # }


        self.ch_embedder = torch.nn.Embedding(chnls, 512)

        # [batch x  (hidden_size * 512) x 1] ->[batch x (32*hidden_size) x 1]  repeating time_reduced times
        self.multichanel_representation = torch.nn.Sequential(
            # TransposeCustom(),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, dilation=1, stride=2, padding=1),   #hidden_size * 512 -> hidden_size * 256
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, dilation=1, stride=2, padding=1),   #hidden_size * 256->hidden_size * 128
            torch.nn.GELU(),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, dilation=1, stride=2, padding=1),   #hidden_size * 128 -> hidden_size * 64
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, dilation=1, stride=2, padding=1),   #hidden_size * 64->hidden_size * 32
            torch.nn.GELU()
        )



        # # [batch x time_reduced x hidden_size x 512] ->[batch x time_reduced x (ch*hidden_size)]
        # self.multichanel_representation = torch.nn.Sequential(
        #     torch.nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4, dilation=1, stride=2, padding=1),   #128->64
        #     torch.nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=4, dilation=1, stride=2, padding=1),   #64->32
        #     torch.nn.GroupNorm(num_groups = 1024 // 2, num_channels = 1024),
        #     torch.nn.GELU(),
        #     torch.nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=4, dilation=1, stride=2, padding=1), # 32->16
        #     torch.nn.Conv1d(in_channels=2048, out_channels=4096, kernel_size=4, dilation=1, stride=2, padding=1), # 16->8
        #     torch.nn.GroupNorm(num_groups=4096 // 2, num_channels=4096),
        #     torch.nn.GELU(),
        #     torch.nn.Conv1d(in_channels=4096, out_channels=8192, kernel_size=4, dilation=1, stride=2, padding=1), # 8->4
        #     torch.nn.Conv1d(in_channels=8192, out_channels=16384, kernel_size=4, dilation=1, stride=2, padding=1), # 4->2
        #     torch.nn.GroupNorm(num_groups=16384 // 2, num_channels=16384),
        #     torch.nn.GELU(),
        #     torch.nn.Conv1d(in_channels=16384, out_channels=32768, kernel_size=2, dilation=1, stride=1, padding=0), # 2->1
        #     torch.nn.Conv1d(in_channels=32768, out_channels=chnls * hidden_size, kernel_size=1, dilation=1, stride=1, padding=0),# 1->1
        #     torch.nn.GroupNorm(num_groups=chnls * hidden_size // 2, num_channels=chnls * hidden_size),
        #     torch.nn.GELU(),
        # )


        self.time_self_attention_embedder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=chnls, out_channels=32, kernel_size=3, dilation=1, stride=1, padding=1),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, dilation=1, stride=1),
            torch.nn.ReLU6(),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=1, stride=1, padding=1),
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, dilation=1, stride=1),
            torch.nn.ReLU6(),
            torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, dilation=1, stride=1, padding=1),
            torch.nn.Conv1d(in_channels=512, out_channels=hidden_size, kernel_size=1, dilation=1, stride=1),
            torch.nn.ReLU6(),
            TransposeCustom(),
        )

        self.norm = torch.nn.LayerNorm(hidden_size)
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dropout(attention_probs

    def forward(self, x, ch_vector):

        embedding= self.embedder(x)           # input [batch x time x ch ]  output [batch x time_reduced x hidden_size x ch]

        embedding = torch.transpose(embedding, 1, 2)

        self.time_reduced = embedding.shape[1]

        embedding = torch.reshape(embedding, (embedding.shape[0], embedding.shape[1], self.hidden_size, self.chnls))

        ch_embedding = self.ch_embedder(ch_vector)
        ch_embedding = ch_embedding[:,None,:,:]
        # torch.reshape(ch_embedding,(1,1,4,22,512))
        # torch.reshape(ch_embedding,(1,1,ch_embedding.shape[0],ch_embedding.shape[1],ch_embedding.shape[2]))
        # weighted_chanels_embedding = torch.matmul(embedding, ch_embedding) #  [batch x time_reduced x hidden_size x ch] ->  [batch x time_reduced x hidden_size x 512]
        weighted_chanels_embedding = torch.einsum('abik, adkj -> abij', embedding, ch_embedding)
        # logic - multiple each chanel to weigth vector[512] and sum it. Here main beamforming logic how from multi chanels move to 1 chanel for each pseudo spectr [hidden_size]
        # On this step we could check how each chanel envolved in next calculation.
        # On next step we just resize [hidden_size x 512] to [hidden_size*ch,1]  using cnn, thats help to hold connection of close parts in pseudo spectr  [hidden_size]

        weighted_chanels_embedding = torch.reshape(weighted_chanels_embedding, (weighted_chanels_embedding.shape[0], weighted_chanels_embedding.shape[1], self.hidden_size * 512))
        # [batch x time_reduced x hidden_size x 512] ->[batch x time_reduced x (hidden_size * 512)]
        # it packed by cat(512, 512, ... 512) so it is stil pseudo spectr with demention hidden_size * 512
        cnn_emb = torch.zeros((weighted_chanels_embedding.shape[0],weighted_chanels_embedding.shape[1],self.hidden_size * 32)).to(weighted_chanels_embedding.device)
        # cnn_emb will be pseudo spectr with demention hidden_size * 32

        for i in range(self.time_reduced):
            cnn_emb[:,i,:][:,None,:] = self.multichanel_representation(weighted_chanels_embedding[:,i,:][:,None,:])

        # cnn_emb = self.multichanel_representation(weighted_chanels_embedding) # [batch x time_reduced x (hidden_size * 512)] ->[batch x time_reduced x (ch*hidden_size)]

        return cnn_emb

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


class EEGEmbedder(torch.nn.Module):
    def __init__(self, hidden_size=128, chnls=len(mitsar_chls)):
        super(EEGEmbedder, self).__init__()
        self.input_embedder = InputEmbedder(hidden_size, chnls)
        self.hidden_size = hidden_size
        self.chnls = chnls
        self.output_size = hidden_size * 32

        config = BertConfig(is_decoder=False,
                            add_cross_attention=False,
                            ff_layer='linear',
                            hidden_size=self.output_size,
                            num_attention_heads=8,
                            num_hidden_layers=4,
                            conv_kernel=1,
                            conv_kernel_num=1)
        self.model = BertEncoder(config)

        self.mask_embedding = torch.nn.Parameter(torch.normal(0, self.output_size ** (-0.5), size=(self.output_size,)),
                                                 requires_grad=True)

        self.classification = torch.nn.Sequential(
            torch.nn.Linear(self.output_size, 1),
        )

        self.upconvolution = torch.nn.Sequential(
            TransposeCustom(),
            torch.nn.ConvTranspose1d(in_channels=self.output_size, out_channels=512, kernel_size=11, stride=1,
                                     padding=5),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose1d(in_channels=512, out_channels=chnls * 10, kernel_size=9,  stride=1,
                                     padding=4),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose1d(in_channels=chnls * 10, out_channels=512, kernel_size=9,  stride=1,
                                     padding=4),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose1d(in_channels=512, out_channels=self.output_size, kernel_size=7,  stride=1,
                                     padding=3),
            torch.nn.ReLU6(),
            TransposeCustom(),
            torch.nn.LayerNorm(self.output_size)
        )

        self.pos_encoder = PositionalEncoding(self.output_size)

        self.eeg_classification = torch.nn.Sequential(
            torch.nn.Linear(self.output_size, 1),
        )
        self.eeg_classification2 = torch.nn.Sequential(
            torch.nn.Linear(self.output_size, 2),
        )


    def forward(self, current, negative, channels, attention_mask, pretrain = True):
        # current  [batch x time x ch]
        # negative  [batch x time x ch]

        curent_embeding = self.input_embedder(current, channels)
        negative_embeding = self.input_embedder(negative, channels)

        cls_token = torch.zeros((curent_embeding.shape[0], 1, curent_embeding.shape[2])).to(curent_embeding.device)

        ############### embedding.shape[2] have to be even ( div 2 ) !
        end_token = torch.from_numpy(np.full((int(curent_embeding.shape[2] * curent_embeding.shape[0] / 2), 2), [-4, -5]).reshape((curent_embeding.shape[0], 1 , curent_embeding.shape[2]))).to(curent_embeding.device)
        if pretrain:
            border = int(0.2 * curent_embeding.shape[1])
            split_point = np.random.randint(border, curent_embeding.shape[1]-border)

            embedding_before_masking = torch.cat([cls_token, curent_embeding[:,:split_point,:], end_token, negative_embeding[:,:curent_embeding.shape[1] - split_point,:], end_token], 1) # must be curent_embeding.shape[0]  usual  about 80
            embedding_before_masking = self.pos_encoder(embedding_before_masking)

            curent_masked = curent_embeding.clone()
            negative_masked = negative_embeding.clone()

            curent_mask = _make_mask((curent_masked.shape[0], curent_masked.shape[1]), 0.05, curent_masked.shape[1], 5).to(current.device)

            curent_masked[curent_mask] = self.mask_embedding.to(curent_embeding.device)
            negative_mask = _make_mask((curent_masked.shape[0], curent_masked.shape[1]), 0.05, curent_masked.shape[1], 5).to(current.device)
            negative_masked[negative_mask] = self.mask_embedding

            embedding_masked = torch.cat([cls_token, curent_masked[:,:split_point,:], end_token, negative_masked[:,:curent_embeding.shape[1] - split_point,:], end_token], 1) # must be curent_embeding.shape[0]  usual  about 80

            embedding_masked = self.pos_encoder(embedding_masked)

            encoder_output = self.model(embedding_masked, output_hidden_states=True,
                                        output_attentions=True)[0]

            negative_predict = self.classification(encoder_output)[:, 0]

            decoded_predict = self.upconvolution(encoder_output)

            return decoded_predict, embedding_before_masking, negative_predict
        else:
            embedding = torch.cat([cls_token, curent_embeding, end_token], 1) # must be curent_embeding.shape[0]  usual  about 80
            embedding = self.pos_encoder(embedding)
            encoder_output = self.model(embedding, output_hidden_states=True,
                                        output_attentions=True)[0]
            classification_output =self.eeg_classification2(encoder_output[:,0,:])

            return classification_output


class EEGEmbedder_old(torch.nn.Module):
    def __init__(self):
        super(EEGEmbedder, self).__init__()
        self.config = BertConfig(is_decoder=False,
                            add_cross_attention=False,
                            ff_layer='linear',
                            hidden_size=512,
                            num_attention_heads=8,
                            num_hidden_layers=8,
                            conv_kernel=1,
                            conv_kernel_num=1)
        self.model = BertEncoder(self.config)

        self.pos_e = PositionalEncoding(512, max_len=6000)
        self.ch_embedder = torch.nn.Embedding(len(mitsar_chls), 512)
        self.ch_norm = torch.nn.LayerNorm(512)

        self.input_norm = torch.nn.LayerNorm(2)
        self.input_embedder = torch.nn.Sequential(
            TransposeCustom(),
            torch.nn.Conv1d(21, 32, 5, 2, padding=0),
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
            torch.nn.Conv1d(512, 512, 1, 1, padding=0),
            torch.nn.GroupNorm(512 // 2, 512),
            torch.nn.GELU(),
            TransposeCustom(),
            # torch.nn.LeakyReLU(),
        )

        self.output_embedder = torch.nn.Linear(512, 512)
        self.transpose = TransposeCustom()

        self.mask_embedding = torch.nn.Parameter(torch.normal(0, 512 ** (-0.5), size=(512,)),
                                                 requires_grad=True)
        self.placeholder = torch.nn.Parameter(torch.normal(0, 512 ** (-0.5), size=(512,)),
                                              requires_grad=True)
        self.classifier = torch.nn.Sequential(
            # torch.nn.Linear(self.config.hidden_size, self.config.num_labels),
            torch.nn.Linear(self.config.hidden_size, 256),
            torch.nn.Linear(256, 2),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs, attention_mask, ch_vector, msk=False, clf=False):
        embedding = self.input_embedder(inputs)
        # create embedding for two channel indexes and sumup them to a single one
        ch_embedding = self.ch_embedder(ch_vector).sum(1)
        ch_embedding = ch_embedding[:, None]
        embedding += ch_embedding
        embedding_unmasked = embedding.clone()  # keep for loss calculation
        # perform masking
        if msk:
            mask = _make_mask((embedding.shape[0], embedding.shape[1]), 0.05, embedding.shape[1], 10)
            embedding[mask] = self.mask_embedding

        # additional vector for classification tasks later
        placeholder = torch.zeros((embedding.shape[0], 1, embedding.shape[2]), device=embedding.device)
        placeholder += self.placeholder
        embedding = torch.cat([placeholder, embedding], 1)
        encoder_output = self.model(embedding, output_hidden_states=True,
                                    output_attentions=True)[0]

        encoder_output = self.output_embedder(encoder_output)

        if clf:
            logits = self.classifier(encoder_output[:, 0])  # sequence classification
            # logits = self.classifier(encoder_output)  # token classification ?
            return logits

        return encoder_output[:, 1:], embedding_unmasked




class EEGClassificator(torch.nn.Module):
    def __init__(self):
        super(EEGClassificator, self).__init__()

        self.EEGEmbedder_model = EEGEmbedder()
        # self.EEGEmbedder_model = torch.load('/home/evgeniy/eeg_processing/models/model_v1.npy')
        # self.EEGEmbedder_model = torch.load('/home/evgeniy/models/EEGEmbeder_TUH_Bert_from_Timur_25_08_2023/step.pt')
        # self.EEGEmbedder_model.load_state_dict(torch.load('/home/evgeniy/models/EEGEmbeder_TUH_Bert_from_Timur_25_08_2023/step_25epochs.pt'))
        # self.EEGEmbedder_model.load_state_dict(torch.load('/media/hdd/evgeniy/eeg_models/step_25epochs.pt'))
        self.EEGEmbedder_model.eval()

        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()

        self.classification = torch.nn.Sequential(
            # torch.nn.Linear(768, 256),
            torch.nn.Linear(512, 256),
            torch.nn.Linear(256, 2),
            torch.nn.Sigmoid()
        )


    def forward(self, inputs, attention_mask, ch_vector,placeholder):
        EEGembedding = self.EEGEmbedder_model(inputs, attention_mask, ch_vector)
        pred = self.classification(EEGembedding)
        return pred, EEGembedding

# ----------------------------------------Data-----------------------------------------------------
class TEST_TUH(torch.utils.data.Dataset):
    def __init__(self, path): #, tuh_filtered_stat_vals):
        super(TEST_TUH, self).__init__()
        self.main_path = path
        self.paths = path
        # self.tuh_filtered_stat_vals = tuh_filtered_stat_vals
        # self.paths = ['{}/{}'.format(self.main_path, i) for i in os.listdir(self.main_path)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx, negative=False):
        path = self.paths[idx]
        # take 60s of recording with specified shift
        key = False
        while (key == False):
            try:
                # sample = np.load(path, allow_pickle=True).item()['value']
                sample = np.load(path, allow_pickle=True).item()
                key = True
            except Exception as e:
                print("Path: {} is broken ".format(path), e)
                path = np.random.choice(self.paths, 1)[0]
                # sample = np.load(path, allow_pickle=True).item()['value']
        real_len = min(3000, sample['value_pure'].shape[0])
        # if np.random.choice([0, 1], p=[0.9, 0.1]):
        #     real_len = np.random.randint(real_len // 2, real_len)

        # sample = torch.from_numpy(sample[:6000].astype(np.float32)).clone()
        # channels_ids = [i for i, val in enumerate(mitsar_chls) if val not in ['FCZ', 'PZ']]
        # channels_ids = [i for i, val in enumerate(sample['channels']) if i != 3 and val in mitsar_chls]
        channels_ids = [i for i, val in enumerate(sample['channels']) if val in mitsar_chls]

        sample = sample['value_pure'][:real_len]

        # choose 2 random channels
        channels_to_train = channels_ids  # np.random.choice(channels_ids, 2, replace=False)
        channels_vector = torch.tensor((channels_to_train))
        sample = sample[:, channels_to_train]

        sample_norm = sample
        # sample_norm = (sample - self.tuh_filtered_stat_vals['mean_vals_filtered'][channels_vector]) / (
        # self.tuh_filtered_stat_vals['std_vals_filtered'][channels_vector])

        # sample_norm = sample_norm * 2 - 1
        # _, mask = masking(sample_norm)
        if sample_norm.shape[0] < 3000:
            sample_norm = np.pad(sample_norm, ((0, 3000 - sample_norm.shape[0]), (0, 0)))



        if np.random.choice([0, 1], p=[0.7, 0.3]) and not negative:
            index = np.random.choice(self.__len__() - 1)
            negative_sample = self.__getitem__(index, True)
            negative_path = negative_sample['path']
            negative_sample_norm = negative_sample['current'].numpy()

            negative_person = negative_sample['path'].split('/')[-1]  # .split('_')
            current_person = path.split('/')[-1]  # .split('_')
            if negative_person.split('_')[0] == current_person.split('_')[0] and \
                    abs(int(negative_person.split('_')[1][:-4]) - int(current_person.split('_')[1][:-4])) < 20000:
                negative_label = torch.tensor(0)               # возможно стоит запретить позитивы отличающиеся < 20000 , если состояние реально изменилось то сеть будет учиться странному.
            else:
                negative_label = torch.tensor(1)
        else:
            negative_sample_norm = sample_norm.copy()
            negative_label = torch.tensor(0)
            negative_path = ''

        attention_mask = torch.ones(3000)
        attention_mask[real_len:] = 0
        return {'current': torch.from_numpy(sample_norm).float(),
                'negative': torch.from_numpy(negative_sample_norm).float(),
                'path': path,
                'label': negative_label,
                'channels': channels_vector,
                'attention_mask': attention_mask}
class TEST_TUH_old(torch.utils.data.Dataset):
    def __init__(self, path):
        super(TEST_TUH_old, self).__init__()
        self.main_path = path
        self.paths = path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # take 60s of recording with specified shift
        key = False
        while (key == False):
            try:
                sample = np.load(path, allow_pickle=True).item()
                key = True
            except Exception as e:
                print("Path: {} is broken ".format(path), e)
                path = np.random.choice(self.paths, 1)[0]

        signal = sample['value_pure']
        real_len = signal.shape[0]
        channels_ids = [i for i, val in enumerate(sample['channels']) if i != 3 and val in mitsar_chls]

        # choose 2 random channels
        # channels_to_train = np.random.choice(channels_ids, 2, replace=False)
        # channels_vector = torch.tensor((channels_to_train))

        # use all available
        channels_to_train = channels_ids
        channels_vector = torch.tensor((channels_to_train))
        signal = signal[:, channels_to_train]

        # remove normalization for now with within channel z-norm
        # sample_norm = (sample - tuh_filtered_stat_vals['min_vals_filtered'][channels_vector]) / (tuh_filtered_stat_vals['max_vals_filtered'][channels_vector] - tuh_filtered_stat_vals['min_vals_filtered'][channels_vector] + 1e-6)
        sample_norm_mean = signal.mean()
        sample_norm_std = np.std(signal)

        signal_norm = (signal - sample_norm_mean) / (sample_norm_std)

        if signal_norm.shape[0] < 6000:
            signal_norm = np.pad(signal_norm, ((0, 6000 - signal_norm.shape[0]), (0, 0)))

        attention_mask = torch.ones(6000)
        attention_mask[real_len:] = 0
        return {'anchor': torch.from_numpy(signal_norm).float(),
                # 'label': sample_label,
                # 'anchor_masked': torch.from_numpy(sample_masked).float(),
                # 'mask': torch.tensor(mask),
                'channels': channels_vector,
                'attention_mask': attention_mask}


def load_HSE_stage2_and_train_test_splitting():
    exclude_user = [10]
    exclude_demo_user = [310]
    server_34 = '/media/hdd/data/HSE_math_exp_2/processing_internal.v2/processing_internal.v2'
    server_56 = '/home/data/HSE_math_exp_2/processing_internal.v2'

    all_paths = ['/media/hdd/data/HSE_math_exp_2/processing_internal.v2/processing_internal.v2/sliced.limited_ch/' + \
                 j for j in os.listdir('/media/hdd/data/HSE_math_exp_2/processing_internal.v2/processing_internal.v2/sliced.limited_ch') if '.npy' in j and 'y.npy' not in j]

    train_paths_X = []
    train_excl_paths_X = []
    test_paths_X = []
    test_excl_paths_X = []
    test_excl_demo_paths_X = []

    train_paths_Y = []
    train_excl_paths_Y = []
    test_paths_Y = []
    test_excl_paths_Y = []
    test_excl_demo_paths_Y = []

    for user in range(23, 104):
        user_paths_x = [i for i in all_paths if int(i.split('/')[-1].split('_')[1]) == user]
        user_paths_x = sorted(user_paths_x, key=lambda x: int(x.split('/')[-1].split('_')[2].split('.')[0]))
        user_paths_y = [(server_34 + '/sliced.limited_ch/{}_y.npy').format(user)] * len(
            user_paths_x)

        user_paths = list(zip(user_paths_x, user_paths_y))

        length = len(user_paths)
        if user in exclude_user:
            continue
        #     train_excl_paths_X.extend(user_paths[:int(length * 0.66)])
        #     test_excl_paths_X.extend(user_paths[int(length * 0.66):])
        # else:
        else:
            t1 = user_paths[:int(length * 0.66)][::3]                    # чтобы ослабить переобучение делим на 3
            # t1 = user_paths[:int(length * 0.66)]
            train_paths_X.extend(t1)
            test_paths_X.extend(user_paths[int(length * 0.66):])

    for user in exclude_user:
        user_paths_x = [i for i in all_paths if int(i.split('/')[-1].split('_')[1]) == user]
        user_paths_x = sorted(user_paths_x, key=lambda x: int(x.split('/')[-1].split('_')[2].split('.')[0]))
        user_paths_y = [(server_34+'/sliced.limited_ch/{}_y.npy').format(user)] * len(
            user_paths_x)

        user_paths = list(zip(user_paths_x, user_paths_y))

        length = len(user_paths)
        train_excl_paths_X.extend(user_paths[:int(length * 0.5)])
        test_excl_paths_X.extend(user_paths[int(length * 0.5):])

    for user in exclude_demo_user:
        user_paths_x = [i for i in all_paths if int(i.split('/')[-1].split('_')[1]) == user]
        user_paths_x = sorted(user_paths_x, key=lambda x: int(x.split('/')[-1].split('_')[2].split('.')[0]))
        user_paths_y = [(server_34+'/sliced.limited_ch/{}_y.npy').format(user)] * len(
            user_paths_x)

        user_paths = list(zip(user_paths_x, user_paths_y))

        length = len(user_paths)
        test_excl_demo_paths_X.extend(user_paths)

    return train_paths_X, test_paths_X, train_excl_paths_X, test_excl_paths_X

from sklearn import preprocessing




class TEST(torch.utils.data.Dataset):
    def __init__(self, paths):
        super(TEST, self).__init__()
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        channels_selected_indexes = [HSE_Stage2_channels.index(i) for i in HSE_Stage2_channels]                     #  из обучения на датасете HSE Stage2

        # # channels_ids = [m_channels.index(val) if val in m_channels else -1 for i, val in enumerate(TUH_chanels)]  # sample['channels']  От тимура
        # # channels_ids = [HSE_Stage2_channels.index(val) if val in HSE_Stage2_channels else -1 for i, val in enumerate(mitsar_chls)]    # sample['channels'] предположительно дополненые -1 каналы митцара которые пересеклись с HST Stage2
        # channels_ids = [HSE_Stage2_channels.index(val) if val in HSE_Stage2_channels else -1 for i, val in enumerate(TUH_chanels_for_training)]    # каналы датасета дополненые -1 до каналов претрейна
        # channels_ids_with_3 = [HSE_Stage2_channels.index(val) if val in HSE_Stage2_channels else -1 for i, val in enumerate(TUH_chanels_for_training_plus3)]    # каналы датасета дополненые -1 до каналов претрейна
        # channels_ids_on_TUH = [i for i, val in enumerate(TUH_chanels) if i != 3 and val in mitsar_chls]             # гипотеза как взять пересечение TUH vs mitsar чтобы повторить в точности пайплайн из претрейна
        # channels_HSE_St2_vs_TUH = [i for i, val in enumerate(HSE_Stage2_channels) if val in TUH_chanels_for_training]             #пересечение HSE_Stage2_channels vs TUH_chanels_for_training


        #            -    ------------- претрейн теперь работает с 22 каналами mitsar_chls    --------------------------
        channels_ids = [HSE_Stage2_channels.index(val) if val in HSE_Stage2_channels else 0 for i, val in enumerate(mitsar_chls)]    # каналы датасета дополненые -1 до каналов претрейна


        path = self.paths[idx]
        data = np.load(path[0])
        data = np.clip(preprocessing.scale(data[:, channels_selected_indexes]), -3, 3)
        slice_num = int(path[0].split('/')[-1].split('_')[2].split('.')[0])

        label = np.load(path[1])[slice_num]
        # 2 and 1 labels in one class
        if label == 2:
            label = 1

        # use all available
        channels_to_train = channels_ids
        channels_vector = torch.tensor((channels_to_train))
        signal = data[:, channels_to_train]
        # signal = signal[:, channels_to_train]

        signal[:,np.where(channels_vector == -1)[0]] *= 0   #  сперва мусор в пустых каналах, а теперь зануляем, можно как вариант скопировать или заинтерпалировать / собственно сейчас туда О2 попадает, обсудить с Рафом

        input_channels_vector = torch.tensor(channels_ids)


        # remove normalization for now with within channel z-norm
        # sample_norm = (sample - tuh_filtered_stat_vals['min_vals_filtered'][channels_vector]) / (tuh_filtered_stat_vals['max_vals_filtered'][channels_vector] - tuh_filtered_stat_vals['min_vals_filtered'][channels_vector] + 1e-6)
        sample_norm_mean = signal.mean()
        sample_norm_std = np.std(signal)

        signal_norm = (signal - sample_norm_mean) / (sample_norm_std)

        # channels = channels_vector.clone().detach()
        # sample = butter_bandpass_filter_v2(sample, 1, 40, 256)
        # sample = torch.from_numpy(sample[:3000].astype(np.float32)).clone()
        # ВОТ СКОЛЬКО НАРЕЗАТЬ И НАДО ЛИ ПАДИТЬ БОЛЬШОЙ ВОПРОС!!!
        sample = torch.from_numpy(signal_norm[:60000].astype(np.float32)).clone()
        return {'anchor': sample,
                'label': torch.tensor(label).long(),  # == sample_label, ?
                # 'channels': channels_vector_wuth_3  #channels_vector
                'channels': input_channels_vector
                #         'correct': sample_correct,
                #         'pure_label': torch.tensor(self.label[idx])
                }

# ---------------------------------Training-----------------------------------------------------------------------
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    np.random.seed((torch_seed + worker_id) % 2 ** 30)

from torch.optim.lr_scheduler import _LRScheduler
class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, d_model=512, last_NoamLR_epoch = -1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        # print(NoamLR.get_lr() , last_epoch)
        self.last_NoamLR_epoch = last_NoamLR_epoch
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < 1:
            last_epoch = max(self.last_NoamLR_epoch,1)
            last_epoch = max(last_epoch, self.last_epoch)
            self.last_epoch = last_epoch
        else:
            last_epoch = self.last_epoch
        factor = min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        # scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        # return [base_lr * scale for base_lr in self.base_lrs]
        return [base_lr * self.d_model ** (-0.5) * factor for base_lr in self.base_lrs]
    def last_epoch_cust(self):
        return self.last_epoch


def _calculate_similarity( z, c, negatives):
    c = c[..., :].permute([0, 2, 1]).unsqueeze(-2)
    z = z.permute([0, 2, 1]).unsqueeze(-2)

    # In case the contextualizer matches exactly, need to avoid divide by zero errors
    negative_in_target = (c == negatives).all(-1)
    targets = torch.cat([c, negatives], dim=-2)

    logits = torch.nn.functional.cosine_similarity(z, targets, dim=-1) / 0.1
    if negative_in_target.any():
        logits[1:][negative_in_target] = float("-inf")

    return logits.view(-1, logits.shape[-1])

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

def pretrain_BERT_TUH(pretrain_path_save, last_step, continue_train = False, last_lt_scheduler_step_number = 0):
    # model = EEGClassificator()

    #  -------------------          train BERT TUH  -----------------------------------------------------
    model = EEGEmbedder()
    if continue_train:
        model.load_state_dict(torch.load(last_step), strict=False)

    splitted_paths = ['/media/hdd/data/TUH_pretrain.filtered_1_40.v2.splited/{}'.format(i) for i in
                      os.listdir('/media/hdd/data/TUH_pretrain.filtered_1_40.v2.splited/')]

    # tuh_filtered_stat_vals = np.load('/home/data/TUH_pretrain.filtered_1_40/stat_vals.npy', allow_pickle=True).item()

    batch_sz = 16
    train_dataset = TEST_TUH(splitted_paths[:-15000]) #,tuh_filtered_stat_vals)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True, worker_init_fn=worker_init_fn)

    test_dataset = TEST_TUH(splitted_paths[-15000:]) #,tuh_filtered_stat_vals)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=1, drop_last=True,
                                              worker_init_fn=worker_init_fn)

    # batch_size = 128,  (у Тимура 64)
    # num_workers = 1    (у Тимура 0)


    model.train()

    lr_d = 1    #1e-6   # У Тимура иногда стоит 1 !!!
    acc_size = 8

    training_epochs1 = 1000000 // len(train_loader)
    # optim = torch.optim.AdamW(model.classifier.parameters(), lr=lr_d, weight_decay=1)
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    model_test = torch.nn.DataParallel(model)
    model_test.to('cuda:0')

    # model_test = model
    # model_test.to(device)

    writer = SummaryWriter('logs')

    # loss_func = torch.nn.MSELoss()

    if continue_train:
        scheduler = NoamLR(optim, 100000, 512, last_NoamLR_epoch=last_lt_scheduler_step_number)
    else:
        scheduler = NoamLR(optim, 100000, 512)
    print('scheduler.last_epoch:',scheduler.last_epoch_cust(), 'scheduler.get_lr():', scheduler.get_lr())
    loss_func = torch.nn.BCEWithLogitsLoss()

    # print('before train,  test metrics:', end="")
    # max_b_acc_test = short_check_results(model_test, test_loader)   # label отсутствуют - поэтому метрики не посчитать

    if continue_train:
        der = 0
        try:
            with torch.no_grad():
                for batch in test_loader:
                    decoded_predict, embedding_before_masking, negative_predict = model_test(
                        batch['current'],  #.to(device)
                        batch['negative'],
                        batch['channels'],
                        batch['attention_mask'],
                        True
                    )
                    loss1 = torch.nn.CrossEntropyLoss()(decoded_predict, embedding_before_masking)
                    loss2 = loss_func(negative_predict.view(-1), batch['label'].float().to('cuda:0'))

                    loss = loss1 + loss2

                    loss = loss.mean() / acc_size
                    der += loss
            der /= len(test_loader)
            writer.add_scalar('Loss/test', der, 0)
            print('continue_train:',continue_train, 'Loss: {}\t'.format(der))
        except:
            raise

    plt_train_loss_list1 = []
    plt_test_loss_list1 = []
    test_bacc_history = []
    train_bacc_history = []
    print('start training, steps each epoch == len(train_loader): ', len(train_loader), 'Batzh size:', batch_sz, ' training_epochs1:',training_epochs1)


    for epoch in range(25, training_epochs1):
        steps = 0
        train_loss_list1 = []
        test_loss_list1 = []
        mean_loss = 0
        acc_step = 0
        for batch in train_loader:
            decoded_predict, embedding_before_masking, negative_predict = model_test(
                batch['current'],
                batch['negative'],
                batch['channels'],
                batch['attention_mask'],
                True
            )

            loss1 = torch.nn.CrossEntropyLoss()(decoded_predict, embedding_before_masking)
            loss2 = loss_func(negative_predict.view(-1), batch['label'].float().to('cuda:0'))

            loss = loss1 + loss2

            loss = loss.mean()
            loss.backward()
            mean_loss += loss.item()
            acc_step += 1
            steps += 1
            # if steps > 20:
            #     break
            # raise
            if acc_step != 0 and acc_step % acc_size == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad()
                if steps % 100 >= 0:
                    print('Loss/train\t{}'.format(mean_loss / acc_size),'steps:',steps, 'scheduler.last_epoch:',scheduler.last_epoch_cust(), 'Lr:', scheduler.get_lr())
                writer.add_scalar('Loss/train', mean_loss / acc_size, steps, scheduler.last_epoch_cust(), scheduler.get_lr())
                train_loss_list1.append(mean_loss / acc_size)
                mean_loss = 0

            if steps != 0 and steps % 1000 == 0:
                der = 0
                try:
                    with torch.no_grad():
                        for batch in test_loader:
                            decoded_predict, embedding_before_masking, negative_predict = model_test(
                                batch['current'],
                                batch['negative'],
                                batch['channels'],
                                batch['attention_mask'],
                                True
                            )
                            loss1 = torch.nn.CrossEntropyLoss()(decoded_predict, embedding_before_masking)
                            loss2 = loss_func(negative_predict.view(-1), batch['label'].float().to('cuda:0'))

                            loss = loss1 + loss2

                            loss = loss.mean() / acc_size
                            der += loss
                    der /= len(test_loader)
                    writer.add_scalar('Loss/test', der, steps,scheduler.last_epoch_cust(), scheduler.get_lr())

                    print('Loss: {}\t'.format(der))
                    test_loss_list1.append(der)

                except:
                    raise
                Path = pretrain_path_save + '/pretrain_model_ep' +str(epoch) + '_step' + str(steps) + '_of_'+ str(int(len(train_loader))) + '.pt'
                torch.save(model_test.module.state_dict(), Path)
                print('model stored last_step', Path)

        writer.add_scalar('Loss/Train_epoch_avarage', np.mean(train_loss_list1), epoch)
        writer.add_scalar('Loss/Test_epoch_avarage', np.mean(([i.cpu() for i in test_loss_list1])), epoch)

        plt_train_loss_list1.append(np.mean(train_loss_list1))
        plt_loss = np.array([i for i in plt_train_loss_list1])
        plt.plot(plt_loss, label='Train loss')
        plt.show()
        plt_test_loss_list1.append(np.mean(([i.cpu() for i in test_loss_list1])))
        plt_test_loss = np.array([i for i in plt_test_loss_list1])
        plt.plot(plt_test_loss, label='Train loss')
        plt.show()

    return model_test, test_loader
def train_classification_eeg_short(pretrain_path_load, classification_model_path_save):
#  -------------------          train classifier -----------------------------------------------------
#     model = torch.load('/media/hdd/evgeniy/eeg_models/Classification_model_epoch100.npy')
#     Path = '/media/hdd/evgeniy/eeg_models/Classification_model_epoch' + str(101) + '.pt'
#     torch.save(model.module.state_dict(), Path)
#      pretrain_path_load = '/media/hdd/evgeniy/eeg_models/Classification_model_epoch101.pt'

    model = EEGEmbedder()
    model.load_state_dict(torch.load(pretrain_path_load), strict=False)

    train_paths_X, test_paths_X, train_excl_paths_X, test_excl_paths_X = load_HSE_stage2_and_train_test_splitting()
    train_dataset = TEST(train_paths_X)
    test_dataset = TEST(test_paths_X)


    batch_sz = 16
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=1
    )

    model.train()


    lr_d1 = 1e-4
    lr_d2 = 1e-4


    training_epochs1 = 1000000 // len(train_loader)
    optim1 = torch.optim.AdamW(model.model.parameters(), lr=lr_d1, weight_decay=1)
    optim2 = torch.optim.AdamW(model.eeg_classification2.parameters(), lr=lr_d2, weight_decay=1)
    # optim = torch.optim.AdamW(model.eeg_classification2.parameters(), lr=lr_d, weight_decay=1)
    # optim = torch.optim.AdamW(model.parameters(), lr=lr_d, weight_decay=1)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.model.parameters():
        param.requires_grad = True
    for param in model.eeg_classification2.parameters():
        param.requires_grad = True

    model_test = torch.nn.DataParallel(model)
    model_test.to('cuda:0')

    # model_test = model
    # model_test.to(device)
    # model_test.to('cpu')

    loss_func = torch.nn.CrossEntropyLoss()
    # print('before train,  test metrics:', end="")
    max_b_acc_test = 0
    # max_b_acc_test = short_check_results(model_test, test_loader)
    plt_train_loss_list1 = []
    test_bacc_history = []
    train_bacc_history = []
    print('start training')
    current_time = int(round(time.time() * 1000)) # in millisec

    for epoch in range(training_epochs1):
        train_loss_list1 = []
        # ii = 0
        count_0 = 0
        count_1 = 0
        for batch in train_loader:
            # print("iteration:", ii)
            # ii += 1
            # batch = train_dataset.__getitem__(i)
            # optim1.zero_grad()
            optim2.zero_grad()
            count_0 += np.unique(batch['label'].numpy(), return_counts=True)[1][0]
            if np.unique(batch['label'].numpy(), return_counts=True)[1].shape[0] > 1:
                count_1 += np.unique(batch['label'].numpy(), return_counts=True)[1][1]
            # placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5

            ae = model_test(
                batch['anchor'],
                batch['anchor'],
                batch['channels'].long(),
                None,
                False)

            loss = loss_func(ae.view(-1, 2), batch['label'].to('cuda:0').long())
            # predict = model(batch['eeg'].cuda())
            # loss = loss_func(predict, batch['label'].cuda())            # из модели обученой на TUH надо вытащить предикт по разнице вероятностей (см, тестирование и расчет метрик)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_test.parameters(), 1.0)
            mean_loss = loss.item()
            train_loss_list1.append(mean_loss)
            optim1.step()
            optim2.step()

        new_current_time = int(round(time.time() * 1000)) # in millisec
        dt = new_current_time - current_time
        current_time = new_current_time

        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list1)),'count_0:', count_0, 'count_1:',count_1, "Time:", dt/1000)
        plt_train_loss_list1.append(np.mean(train_loss_list1))
        if epoch % 5 == 2:
            print('max:',max_b_acc_test,'test: ', end="")
            test_acc = short_check_results(model_test, test_loader)
            if test_acc > max_b_acc_test:
                max_b_acc_test = test_acc
            test_bacc_history.append(test_acc)
            print('train: ', end="")
            train_bacc = short_check_results(model_test, train_loader)
            train_bacc_history.append(train_bacc)
            # Path = '/media/hdd/evgeniy/eeg_models/Classification_model_epoch' + str(epoch) + '.npy'
            # torch.save(model_test, Path)

            Path = '/media/hdd/evgeniy/eeg_models/Classification_model_epoch' + str(epoch) + '_test_bacc_'+ str(int(100*test_acc)) +'.pt'
            torch.save(model_test.module.state_dict(), Path)

            plt_loss = np.array([i for i in plt_train_loss_list1])
            plt_acc_test = np.array([i for i in test_bacc_history])
            plt_acc_train = np.array([i for i in train_bacc_history])

            plt.plot(plt_loss, label='Train loss')
            plt.show()
            plt.plot(plt_acc_test, label='Test bacc')
            plt.plot(plt_acc_train, label='Train bacc')
            plt.show()
    return model_test, test_loader

def short_check_results(model, test_loader):
    from sklearn.metrics import balanced_accuracy_score
    model = model.eval()

    preds = []
    reals = []
    with torch.no_grad():
        # ii = 0
        for batch in test_loader:
            # print("iteration v:", ii)
            # ii += 1
            # placeholder = torch.zeros((batch['anchor'].shape[0], 1, 512)) - 5
            ae = model(
                batch['anchor'],
                batch['anchor'],
                batch['channels'].long(),
                None,
                False)

            y_pred = (ae[:, 1] > ae[:, 0])
            reals.extend(batch['label'])
            preds.extend(y_pred)

        # y_pred = test_dataset.argmax(-1))[2][0] + +precision_recall_fscore_support(reals, preds.argmax(-1))[2][1]) / 2
    reals = np.array([i.tolist() for i in reals])
    preds = np.array([i.tolist() for i in preds])

    acc = np.sum(preds == reals) / preds.shape[0]  # Не сбалансированная!!
    all_scores = precision_recall_fscore_support(preds, reals)
    balensed_acc = balanced_accuracy_score(reals, preds)
    print('balensed_acc:', balensed_acc, 'acc:',acc, 'f1 precision, recall, fscore ,amount:', all_scores)
    return balensed_acc

def main():
    # print('Hello')
    pretrain_path_save = '/media/hdd/evgeniy/eeg_models'
    # pretrain_path_load = '/media/hdd/evgeniy/eeg_models/step_25epochs.pt'
    pretrain_path_load = '/media/hdd/evgeniy/eeg_models/Classification_model_epoch7_test_bacc_50.pt'
    last_step = '/media/hdd/evgeniy/eeg_models/pretrain_model_ep25_step5000_of_573.pt'
    classification_model_path_save = '/media/hdd/evgeniy/eeg_models/Classification_model_v1.npy'
    # classification_model_path_load = '/home/evgeniy/eeg_processing/models/Classification_model_v2.npy'
    classification_model_path_load = '/media/hdd/evgeniy/eeg_models/Classification_model_v2.npy'
    last_lt_scheduler_step_number = int((6000 + 9169 * (38-25)) / 8 ) # 38 я эпоха - стартуем мы с 25 так повелось, 9169  шага на эпоху, +6000 текущих шагов.
    # т.к. мы делаем backword и scheduler step  только каждый 8й шаг то все делим на 8
    # model, test_loader = pretrain_BERT_TUH(pretrain_path_save, last_step, True, last_lt_scheduler_step_number = last_lt_scheduler_step_number)

    # print('pretrain results:',end="")
    # short_check_results(model, test_loader)
    print('START CLASSIFICATION')
    model, test_loader = train_classification_eeg_short(pretrain_path_load, classification_model_path_save)

    # test_dataset = TEST(test_data, test_label, test_mean_std, HSE_chls, test_correct)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True,
    #                                           worker_init_fn=worker_init_fn)
    # model = torch.load(classification_model_path_load)
    # model.eval()

    # print('classification results:',end="")
    # short_check_results(model, test_loader)



if __name__ == '__main__':
    # a = torch.from_numpy(np.full((int(5*4/2), 2), [1, 2]).reshape((5,1,4)))
    # print(a)
    main()
