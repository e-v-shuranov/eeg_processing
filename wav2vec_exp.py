# !pip install transformers
# !pip install datasets
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from os.path import dirname, abspath, join
import sys

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
# sys.path.append(join(PROJECT_ROOT, 'src'))
sys.path.append(PROJECT_ROOT)

# load pretrained model
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# processor = Wav2Vec2Processor.from_pretrained("models/wav2vec_small_960h.pt")
processor = Wav2Vec2Processor.from_pretrained("/home/evgeniy/models/wav2vec2-base-960h")
# processor = Wav2Vec2Processor.from_pretrained("/home/evgeniy/audio_datasets/models/wav2vec_small_960h.pt")
model = Wav2Vec2ForCTC.from_pretrained("/home/evgeniy/models/wav2vec2-base-960h")


# librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
librispeech_samples_ds = load_dataset("/home/evgeniy/audio_datasets/LibriSpeech/LibriSpeech1", "dev-clean", split="validation")

# input_sample = torch.tensor(librispeech_samples_ds[0])[None, :]

# load audio
# audio_input, sample_rate = sf.read('/home/evgeniy/audio_datasets/LibriSpeech/LibriSpeech1/dev-clean/1272/128104/1272-128104-0000.flac')

audio_input, sample_rate = sf.read((librispeech_samples_ds.data["audio"][0])[1].as_py())

# audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

# INFERENCE

# retrieve logits & take argmax
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
transcription = processor.decode(predicted_ids[0])

# FINE-TUNE

target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"

# encode labels
with processor.as_target_processor():
  labels = processor(target_transcription, return_tensors="pt").input_ids

# compute loss by passing labels
loss = model(input_values, labels=labels).loss
loss.backward()