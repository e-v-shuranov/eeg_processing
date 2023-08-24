import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from dataclasses import dataclass
# from transformers.file_utils import ModelOutput

from typing import Tuple

from typing import Dict, List, Optional, Union
from typing import Any

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from transformers.file_utils import ModelOutput

import torch.nn as nn
from packaging import version

from transformers import (
    Trainer,
    is_apex_available,
)

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

from transformers import EvalPrediction
import torchaudio

from os.path import dirname, abspath, join
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import random


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
        # use_apex


        # loss = self.compute_loss(model, inputs)
        # if self.args.gradient_accumulation_steps > 1:
        #     loss = loss / self.args.gradient_accumulation_steps
        #
        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        # return loss.detach()


        if self.use_cuda_amp:                 #use_cpu_amp?
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_cuda_amp:
            self.scaler.scale(loss.mean()).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.mean().detach()

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
#------------------------------------------------out of main for EMO --------------------------------------------------------------------------
# We need to specify the input and output column
# we need to distinguish the unique labels in our SER dataset

# train_dataset, eval_dataset = Emo_audio_data_upload()
# input_column = "path"
# output_column = "emotion"
# label_list = train_dataset.unique(output_column)
# label_list.sort()  # Let's sort it for determinism
# num_labels = len(label_list)
# print(f"A classification problem with {num_labels} classes: {label_list}")
# # model_name_or_path = "lighteternal/wav2vec2-large-xlsr-53-greek"
# model_name_or_path = "/home/evgeniy/models/wav2vec2-large-xlsr-53-greek"
#
# processor = Wav2Vec2Processor.from_pretrained(model_name_or_path, )
# target_sampling_rate = processor.feature_extractor.sampling_rate
# print(f"The target sampling rate: {target_sampling_rate}")

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate, padding=True)
    result["labels"] = list(target_list)

    return result

import random
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    np.random.seed((torch_seed + worker_id) % 2**30)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, eval_dataset = Emo_audio_data_upload()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                                               drop_last=True,
                                               worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=True,
                                              worker_init_fn=worker_init_fn)

    loss_func = torch.nn.CrossEntropyLoss()
    model_name_or_path = "/home/evgeniy/models/wav2vec2-large-xlsr-53-greek"
    model_bin_path = "/home/evgeniy/models/wav2vec2-large-xlsr-53-greek"

    # model_bin_path = "/home/evgeniy/models/wav2vec_EMO_my_epoch_354"

    from transformers import AutoConfig, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    pooling_mode = "mean"
    output_column = "emotion"
    label_list = train_dataset.unique(output_column)

    # label_list = [0, 1, 2, 3]
    # label_list = [0, 1]                                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! это надо будет вернуть для EEG
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

    model = Wav2Vec2ForSpeechClassification(config=config).to(device)
    model = model.from_pretrained(model_bin_path, config=config).to(device)

    # model = Wav2Vec2ForSpeechClassification.from_pretrained(
    #     model_name_or_path,
    #     config=config,
    # ).to(device)

    #  Пример как разделить инициализацию модели и загрузку весов
    # model = Wav2Vec2ForPreTraining(config)
    # if args.load_from_pretrained is not None:
    #     try:
    #         model = model.from_pretrained(args.model_name_or_path)
    #     except:
    #         print("!!!!! Warning: Pretrained model may not exist. Start training from Scratch")


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

    for epoch in range(0,num_of_epoch):
        train_loss_list = []
        for batch in train_loader:
            optim.zero_grad()
            speech_array = list(map(speech_file_to_array_fn2, batch["path"]))

            features = processor(speech_array, sampling_rate=processor.feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
            input_values = features.input_values.to(device)
            attention_mask = features.attention_mask.to(device)

            output_values = list(map(get_emo_number, batch['emotion']))

            logits = model(input_values, attention_mask=attention_mask).logits
            loss = loss_func(logits,torch.Tensor(output_values).to('cuda:0').long())



            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # нормализация градиентов вместе - уточнить у Тимура профит
            mean_loss = loss.item()
            train_loss_list.append(mean_loss)

            optim.step()
        print('epoch',epoch,'Loss: {}\t'.format(np.mean(train_loss_list)))


if __name__ == '__main__':
    main()