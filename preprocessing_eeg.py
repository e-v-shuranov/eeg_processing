  #  based on Timur :  Math_preproc.train_itself.v5.Batch.256Hz  30/05/2023

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ['http_proxy'] = "http://127.0.0.1:3128"
# os.environ['https_proxy'] = "http://127.0.0.1:3128"


import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os


from scipy import signal

from scipy.signal import resample
from scipy.signal import butter, lfilter
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.signal import freqz

import mne


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


    #-------------------------Main data--------------------------------------------
    # def split_train_test(val, val_labels): size = val.shape[0] train_size = int(size * 0.7)
    #
    # train = val[:train_size]
    # train_labels = val_labels[:train_size]
    # test = val[train_size:]
    # test_labels = val_labels[train_size:]
    #
    # return (train, train_labels), (test, test_labels), train_sizelabels desc:
    # 7 синхронизация ээг и видео
    #
    # 4/44 - начало/конец базовой линии перед экспериментом
    #
    # 11/19 - начало/конец примера пролистывание (смотреть на примеры и ничего не делать)
    #
    # 22/29 - пример flow
    #
    # 33/39 - пример overload
    #
    # 1/10 - эпоха пролистывание
    #
    # 2/20 - эпоха flow
    #
    # 3/30 - эпоха overload
    #---------------------------------------------------------------------------------------------

#-----------------------Precise splitting-------------------------------------------


def split_train_test(val, val_labels, err_labels):
  size = val.shape[0]
  train_size = int(size * 0.7)

  train = val[:train_size]
  train_labels = val_labels[:train_size]
  train_err = err_labels[:train_size]
  test = val[train_size:]
  test_labels = val_labels[train_size:]
  test_err = err_labels[train_size:]

  return (train, train_labels, train_err), (test, test_labels, test_err), train_size

def parse_HSE_exp_data(path, path_meta):
  data2 = mne.io.read_raw_brainvision(path, preload=True)
  # mne filter more clear and nice for borders
  data2 = data2.resample(256)
  # data2 = data2.filter(1, 40)
  meta_data2_pdf = pd.read_excel(path_meta)
  meta_agg = meta_data2_pdf.groupby(['Epoch number']).agg({
      'is_True': list,
      'Epoch': lambda x: list(x)[0]
  }).reset_index()

  meta_dict = {row['Epoch number']: len(row['is_True']) - sum(row['is_True']) if row['Epoch'] != 'scrolling' else -1
               for i, row in meta_agg.iterrows()}
  # data2 = data2.filter(1, 40)
  print(data2.info['ch_names'])

  full_labels = mne.events_from_annotations(data2)[0]

  start_point2 = full_labels[np.where(full_labels[:, 2] == 7)[0][-1]][0]
  end_point2 = full_labels[-1][0]

  data2_pdf = data2.to_data_frame()[['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
                                     'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6',
                                     'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2']]
  print(data2_pdf.columns)

  # start_point2 - moment of experiment start
  data2_np = data2_pdf.values[start_point2:end_point2]

  labels_np = []
  labels_err = []
  labels_correct = []
  i = 0
  stage = 0
  example = 0

  relax_state = None
  while i < len(full_labels):
      if (full_labels[i][2] in [1, 2, 3]):
          relax_state = (full_labels[i][0] - 20 * 256 - start_point2, full_labels[i][0] - start_point2)

          labels_np.append((full_labels[i][0] - 20 * 256 - start_point2, full_labels[i][0] - start_point2, 0,
                            relax_state[0], relax_state[1]))
          labels_err.append((full_labels[i][0] - 20 * 256 - start_point2, full_labels[i][0] - start_point2, 0))
          labels_correct.append((full_labels[i][0] - 20 * 256 - start_point2, full_labels[i][0] - start_point2, 0))
      # if (full_labels[i][2] in [1, 2, 3]):
      #     labels_np.append((full_labels[i][0] - 20 * 256 - start_point2, full_labels[i][0] - start_point2, 0))
      #     labels_err.append((full_labels[i][0] - 20 * 256 - start_point2, full_labels[i][0] - start_point2, 0))
      # print(full_labels[4:][i-1][2])
      # if (full_labels[i][2] in [1, 2, 33]):
      #     labels_np.append((full_labels[i][0] - 20 * 256 - start_point2, full_labels[i][0] - start_point2, 0))
      #     labels_err.append((full_labels[i][0] - 20*256 - start_point2, full_labels[i][0] - start_point2, 0))

      if (full_labels[i][2] in [11, 22, 33]):

          val = full_labels[i][2]
          sp = full_labels[i][0]
          while (full_labels[i][2] not in [19, 29, 39]):
              i += 1

          ep = full_labels[i][0]

          labels_np.append((sp - start_point2, ep - start_point2, val // 10, relax_state[0], relax_state[1]))
          labels_err.append((sp - start_point2, ep - start_point2, meta_dict[stage]))
          labels_correct.append(meta_data2_pdf.is_True[example])
          if (full_labels[i][2] in [1, 2, 3]):
              stage += 1

          sp = full_labels[i][0]
          example += 1

      else:
          i += 1

  result_labels = np.zeros((end_point2))
  result_err = np.zeros((end_point2))
  for sp, ep, val, rsp, rep in labels_np:
      result_labels[sp:ep] = val
  for sp, ep, val in labels_err:
      result_err[sp:ep] = val

  # we flattened labels over origin recording - so we need to cut the same point
  result_labels = result_labels[start_point2:]
  result_err = result_err[start_point2:]
  return data2_np, result_labels, result_err, labels_np, labels_err, labels_correct

def process_exp(path_eeg, path_meta, shuffle=False, traintest_split=False):
  data, labels, labels_err, labels_splited, errors_splited, labels_correct_splitted = parse_HSE_exp_data(path_eeg,
                                                                                                         path_meta)
  data_to_model = []
  relax_data_to_model = []
  label_to_model = []
  correct_to_model = []
  print(len(labels_splited), len(labels_correct_splitted))
  for index, (s, e, l, rs, re) in enumerate(labels_splited):
      sgnl = data[s:e]
      rest_sgnl = data[rs:re]
      rest_sgnl_mean = rest_sgnl.mean(0)
      rest_sgnl_std = rest_sgnl.std(0)
      if (sgnl.shape[0]) > 256 * 3:
          for i in range(0, sgnl.shape[0], 256 * 3):
              lbl = l

              if (sgnl[i:i + 256 * 3].shape[0] == 256 * 3):
                  relax_data_to_model.append((np.concatenate([rest_sgnl_mean[None], rest_sgnl_std[None]])[None]))
                  data_to_model.append(sgnl[i:i + 256 * 3][None])
                  label_to_model.append(lbl)
                  correct_to_model.append(labels_correct_splitted[index])

  data_to_model = np.concatenate(data_to_model)
  relax_data_to_model = np.concatenate(relax_data_to_model)
  label_to_model = np.array(label_to_model)
  correct_to_model = np.array(correct_to_model)

  if shuffle:
      indexes = list(range(len(data_to_model)))
      np.random.shuffle(indexes)
      data_to_model = data_to_model[indexes]
      relax_data_to_model = relax_data_to_model[indexes]
      label_to_model = label_to_model[indexes]
      correct_to_model = correct_to_model[indexes]

  if traintest_split:
      train_signal = data_to_model[:int(len(data_to_model) * 0.7)]
      test_signal = data_to_model[int(len(data_to_model) * 0.7):]

      train_relax = relax_data_to_model[:int(len(data_to_model) * 0.7)]
      test_relax = relax_data_to_model[int(len(data_to_model) * 0.7):]

      train_label = label_to_model[:int(len(data_to_model) * 0.7)]
      test_label = label_to_model[int(len(data_to_model) * 0.7):]

      train_correct_label = correct_to_model[:int(len(data_to_model) * 0.7)]
      test_correct_label = correct_to_model[int(len(data_to_model) * 0.7):]

      return train_signal, train_label, test_signal, test_label, train_correct_label, test_correct_label, train_relax, test_relax

  else:
      return data_to_model, label_to_model, correct_to_model, relax_data_to_model


def main():
    train_signal = []
    train_relax_signal = []
    train_label = []
    train_correct = []
    # test_signal = []
    # test_label = []

    for i in range(2, 23):
        ts, tl, tes, trs = process_exp(
            '/home/data/HSE_math_all/hse_math/{}_math.vhdr'.format(i),
            '/home/data/HSE_math_all/reshenie/Subject{}_math_output.xlsx'.format(i),
            shuffle=False,
            traintest_split=False)

        train_signal.append(ts)
        train_relax_signal.append(trs)
        train_label.append(tl)
        train_correct.append(tes)
        # test_signal.extend(tes)
        # test_label.extend(tel)

    print(train_relax_signal[0].shape)

    # indexes = list(range(len(train_signal)))
    # np.random.shuffle(indexes)
    indexes = np.load('/home/data/HSE_math_all/processed/v3/indexes.non_filtered.precise_split.256Hz.npy')
    train_indexes = indexes[:int(len(indexes) * 0.8)]
    test_indexes = indexes[int(len(indexes) * 0.8):]
    print('train_indexes:',train_indexes,'test_indexes:', test_indexes)
    train_signal_flat = []
    train_label_flat = []
    train_correct_flat = []


    train_mean_std_flat = []
    ranges_ = []

    for i in train_indexes:
        ranges_.append(len(train_signal[i]))
        train_signal_flat.extend(train_signal[i])
        train_label_flat.extend(train_label[i])
        train_correct_flat.extend(train_correct[i])
        train_mean_std_flat.extend(train_relax_signal[i])

    test_signal_flat = []
    test_label_flat = []
    test_correct_flat = []
    test_mean_std_flat = []
    for i in test_indexes:
        test_signal_flat.extend(train_signal[i])
        test_label_flat.extend(train_label[i])
        test_correct_flat.extend(train_correct[i])
        test_mean_std_flat.extend(train_relax_signal[i])

    print(train_correct[i].shape)
    plt.plot(test_label_flat)
    plt.show()
    len(train_signal)

    np.save('/home/evgeniy/eeg_data/v100/train_signal.non_filtered.precise_split.256Hz.npy', train_signal_flat)
    np.save('/home/evgeniy/eeg_data/v100/train_label.non_filtered.precise_split.256Hz.npy', train_label_flat)
    np.save('/home/evgeniy/eeg_data/v100/train_correct.non_filtered.precise_split.256Hz.npy', train_correct_flat)
    np.save('/home/evgeniy/eeg_data/v100/train_mean_std.non_filtered.precise_split.256Hz.npy', train_mean_std_flat)

    np.save('/home/evgeniy/eeg_data/v100/test_signal.non_filtered.precise_split.256Hz.npy', test_signal_flat)
    np.save('/home/evgeniy/eeg_data/v100/test_label.non_filtered.precise_split.256Hz.npy', test_label_flat)
    np.save('/home/evgeniy/eeg_data/v100/test_correct.non_filtered.precise_split.256Hz.npy', test_correct_flat)
    np.save('/home/evgeniy/eeg_data/v100/test_mean_std.non_filtered.precise_split.256Hz.npy', test_mean_std_flat)

# /home/evgeniy/eeg_data
if __name__ == '__main__':
    main()