import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('GPU') # SELECT GPU CONFIG HERE

import json
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import preprocessing

from sklearn.metrics import  precision_recall_fscore_support, balanced_accuracy_score

np.random.seed(42)
# n channels in EEG 
N_CHANNELS = 15
#n classes
N_CLASSES = 3
#n time steps per eeg file  1500 = 3 seconds of eeg 
ECOG_INT_LEN = 1500
#subjects from 25 to 44
subjects = [i for i in range(23, 105)] + [int(os.getenv('USER'))] + [int(os.getenv('USER_FT'))]
# will be exludeded from pretraining and used for finetuning of second model

test_subj = [int(os.getenv('USER'))]
test_subj_ft = [int(os.getenv('USER_FT'))]
print(test_subj)
SPLIT = 0.66 # train/test for first model
TEST_SPLIT = 0.66 if int(os.getenv('USER')) > 20 else 0.5# train/test of excluded subject(s) for second model

#change here for something meaningfull
TEST_N = 1
RUN_NAME = f'eeg2math_test_{TEST_N}'
# Select one of two baseline arhitecture for first model 
TEST_MODEL = 'resnet'
#TEST_MODEL = 'ednet'

# data dir 
DATASET_DIR = os.path.join(os.getcwd(), '/home/data/HSE_math_exp_2/processing_internal.v2/sliced.limited_ch/')
subjects_to_process = [(os.path.join(DATASET_DIR, f'{i}_y.npy'), i) for i in subjects ]
print(subjects_to_process)

# Generating train/test sets 

train_smp = [] # train with all except one
test_smp = [] # test except one

train_smp_s = [] # train of excluded subject
test_smp_s = [] # test of excluded

train_smp_s_ft = [] # train of excluded subject
test_smp_s_ft = [] # test of excluded


tests_sep = [] # # test except one too???

for drd, i in subjects_to_process:
    try:
        tmpY = np.load(drd, allow_pickle=True)
        tmpY = tmpY# - 5 # 5,6,7 labels to 0, 1, 2
        sept = []
        for j in range(tmpY.shape[0]):
            if not os.path.isfile("/home/data/HSE_math_exp_2/processing_internal.v2/sliced.limited_ch/x_{}_{}.npy".format(i, j)):
                # print(i, j)
                continue

            if i in test_subj:
                test_smp_s.append((i,j,tmpY[j]))

            elif i in test_subj_ft:
                # print(j, tmpY.shape[0])
                if j < int(tmpY.shape[0]*TEST_SPLIT):
                        train_smp_s_ft.append((i,j,tmpY[j]))
                else:
                    test_smp_s_ft.append((i,j,tmpY[j]))
            else:
                if j < int(tmpY.shape[0]*SPLIT):

                    if np.random.choice([0, 1], p=[0.8, 0.2]) == 1:
                        train_smp.append((i,j,tmpY[j]))
                else:
                    if np.random.choice([0, 1], p=[0.7, 0.3]) == 1:
                        test_smp.append((i,j,tmpY[j]))
                        sept.append((i,j,tmpY[j]))

        tests_sep.append(sept)
    except Exception as e:
        print(e)
        pass
    
if len(test_smp_s) == 0:
    print(len(test_smp_s), len(train_smp_s))
    print(len(train_smp_s_ft), len(test_smp_s_ft))
    raise
    
print(len(train_smp_s_ft), len(test_smp_s_ft))
print(len(test_smp_s), len(train_smp_s))

def read_one(smpl):
    
    i, j, k = smpl
                         
    y_f = k
    
    x_f = np.load(os.path.join(DATASET_DIR, f"x_{i}_{j}.npy"), allow_pickle=True)
    
    x_f = sklearn.preprocessing.scale(x_f)
    
    x_f = np.clip(x_f, -3, 3)
    
    return x_f.astype("float32"), y_f.astype("int32")
    
def preprocess(idx):
    
    spec, audio = tf.numpy_function(read_one, [idx], [tf.float32, tf.int32])
    
    return spec,  audio


    
train_dataset = tf.data.Dataset.from_tensor_slices((train_smp,))
train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_smp,))
test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
sep_test_ds = []

from tensorflow.keras import layers, Input
from tensorflow import keras


# baseline model resnet style 1d convs converted

def resnet_18_1dconv(input_shape, model_type = 18, use_head = True):
    def res_block(x, n_filters):
        res = x 
        
        x = layers.Conv1D(n_filters, 3, 2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv1D(n_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        res = layers.Conv1D(n_filters, 3, 2, padding="same")(res)
        res = layers.BatchNormalization()(res)
        res = layers.Activation('relu')(res)
        
        x = layers.Add()([x, res])
        
        x = layers.Activation('relu')(x)
        
        return x 
        
    def iden_block(x, n_filters):
        res = x 
        x = layers.Conv1D(n_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv1D(n_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([x, res])
        
        x = layers.Activation('relu')(x)
        
        return x 
    
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv1D(64, 7, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = iden_block(x, 64)
    x = iden_block(x, 64)
    
    x = res_block(x, 128)
    x = iden_block(x, 128)
    
    if model_type > 10:
        x = res_block(x, 256)
        x = iden_block(x, 256)
    
    if model_type > 14:
        x = res_block(x, 512)
        x = iden_block(x, 512)
        
    
    if use_head:
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(N_CLASSES, activation='linear')(x)
    
    else:
        outputs = x
        
    return keras.Model(inputs, outputs, name='res_net')

# 1d for baseline
def resnet_10_lstm_mix(input_shape):
    DOWNSAMPLING =  10
    
    res10 = resnet_18_1dconv(input_shape, model_type = 10, use_head = False)
    inputs = keras.Input(shape=input_shape)

    x = res10(inputs)
    x = x[:,::DOWNSAMPLING,:]
    x = layers.Bidirectional(layers.LSTM(80, return_sequences=True))(x)
    x = layers.Dropout(.1)(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    outputs = layers.Dense(N_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='resnet10_lstm')
    return model



# envelope detector net from paper Petrosyan et al. 2022, Hyperparams selected from internal tests

def ed_net(input_shape, n_branches = 88, lstm_units = 64, filtering_size = 45, envelope_size = 25):
    DOWNSAMPLING =  10
    FILTERING_SIZE = filtering_size
    ENVELOPE_SIZE = envelope_size
    
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv1D(n_branches, 1, padding="same")(inputs)
    x = layers.BatchNormalization(center=False, scale=False)(x)
    
    x = layers.Conv1D(n_branches, FILTERING_SIZE, padding="same", groups=n_branches, use_bias = False)(x)
    x = layers.BatchNormalization(center=False, scale=False)(x)
    x = layers.LeakyReLU(-1)(x)
    
    x = layers.Conv1D(n_branches, ENVELOPE_SIZE, padding="same",  groups=n_branches)(x)
    x = x[:,::DOWNSAMPLING,:]
    x = layers.Bidirectional(layers.LSTM(lstm_units//2))(x)
    x = layers.BatchNormalization(center=False, scale=False)(x)
    outputs = layers.Dense(N_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name=f'ednet_{n_branches}_{lstm_units}_f_{filtering_size}_e_{envelope_size}')
    
    return model

if TEST_MODEL == 'resnet':
    model = resnet_10_lstm_mix((ECOG_INT_LEN, N_CHANNELS))
    
elif TEST_MODEL == 'ednet':
    model = ed_net((ECOG_INT_LEN, N_CHANNELS))


model.summary()

import tensorflow_addons as tfa

weight_decay = 0.00001
LR_RATE = 0.0001 
BATCH_SIZE = 32
N_EPOCH = 25 # 100 - 200  
optimizer = tfa.optimizers.AdamW(learning_rate=LR_RATE, weight_decay=weight_decay)
lfn = tf.keras.losses.SparseCategoricalCrossentropy()
msca = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

checkpoint_filepath = './ckpt/'+model.name+'_'+RUN_NAME+'_best.our_data' + os.getenv('USER') + '.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.compile(optimizer=optimizer, loss= lfn, metrics=msca)

hist = model.fit(train_dataset.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), 
          validation_data = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), 
          epochs=N_EPOCH,
          callbacks=[model_checkpoint_callback]
         )

# loading best weights
model.load_weights('./ckpt/'+model.name+'_'+RUN_NAME+'_best.our_data' + os.getenv('USER') + '.h5')


res = model.evaluate(test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

sep_res = []
for tst in tests_sep:
    if len(tst) == 0:
        continue
    
    sep_test_dataset = tf.data.Dataset.from_tensor_slices((tst,))
    sep_test_dataset = sep_test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    sres = model.evaluate(sep_test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    sep_res.append(sres[1])  

first_model_pretrain_per_user = dict(zip(subjects, sep_res, ))

lmodel = keras.Model(model.inputs, model.layers[-2].output, name=f'lmodel')
lmodel.summary()

# Converting EEG to vectors
train_data = []
train_gts = []

for smp in train_dataset.as_numpy_iterator():
    x, y = smp
    if y != 1:
        train_data.append(np.squeeze(lmodel.predict(np.expand_dims(x, axis=0), verbose=0)))
        train_gts.append(np.clip(y,0,1))

val_data = []
val_gts = []

for smp in test_dataset.as_numpy_iterator():
    x, y = smp
    if y != 1:
        val_data.append(np.squeeze(lmodel.predict(np.expand_dims(x, axis=0), verbose=0)))
        val_gts.append(np.clip(y,0,1))

# Number of vectors in sequence for second model
INTV = 5 

def read_one_train(smpl):   
    y_f = train_gts[smpl]
    x_f = train_data[smpl-INTV:smpl]
    
    return np.array(x_f).astype("float32"), y_f.astype("int32")
    
def lpreprocess_train(idx):
    spec, audio = tf.numpy_function(read_one_train, [idx], [tf.float32, tf.int32])
    return spec,  audio

def read_one_test(smpl):     
    y_f = val_gts[smpl]
    x_f = val_data[smpl-INTV:smpl]
    
    return np.array(x_f).astype("float32"), y_f.astype("int32")
    
def lpreprocess_test(idx):
    spec, audio = tf.numpy_function(read_one_test, [idx], [tf.float32, tf.int32])
    
    return spec,  audio


t_samples = np.arange(INTV, len(train_data))
v_samples = np.arange(INTV, len(val_data))

ltrain_dataset = tf.data.Dataset.from_tensor_slices((t_samples,))
ltrain_dataset = ltrain_dataset.map(lpreprocess_train, num_parallel_calls=tf.data.AUTOTUNE)

ltest_dataset = tf.data.Dataset.from_tensor_slices((v_samples,))
ltest_dataset = ltest_dataset.map(lpreprocess_test, num_parallel_calls=tf.data.AUTOTUNE)

# very simple second model for baseline

def temporal_class_mix(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(128, input_shape[0], padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return keras.Model(inputs , outputs, name=f'temp_bclass_cnn_rnn')
    
    
tmp_model =  temporal_class_mix((INTV, 64)) 
tmp_model.summary()

import tensorflow_addons as tfa

weight_decay = 0.00001
LR_RATE = 0.0001
BATCH_SIZE = 32
N_EPOCH = 25
optimizer = tfa.optimizers.AdamW(learning_rate=LR_RATE, weight_decay=weight_decay)

lfn = tf.keras.losses.BinaryCrossentropy()

msca =  [ 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), 
]

checkpoint_filepath = './ckpt/'+tmp_model.name+'_'+RUN_NAME+'_best.our_data' + os.getenv('USER') + '.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

tmp_model.compile(optimizer=optimizer, loss= lfn, metrics=msca)

hist = tmp_model.fit(ltrain_dataset.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), 
          validation_data = ltest_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), 
          epochs=N_EPOCH,
          callbacks=[model_checkpoint_callback]
         )

tmp_model.load_weights('./ckpt/'+tmp_model.name+'_'+RUN_NAME+'_best.our_data' + os.getenv('USER') + '.h5')
ev_res = tmp_model.evaluate(ltest_dataset.batch(BATCH_SIZE))

from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

preds = tmp_model.predict(ltest_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
second_model_pretrain_metrics = precision_recall_fscore_support(val_gts[INTV:], preds > 0.5, average=None)



# TRAIN SET FOR EXCLUDED FOR FT

# train set of excluded subject
train_dataset1 = tf.data.Dataset.from_tensor_slices((train_smp_s_ft,))
train_dataset1 = train_dataset1.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
# test set of excluded subject but from demo task
test_dataset1 = tf.data.Dataset.from_tensor_slices((test_smp_s,))
test_dataset1 = test_dataset1.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

train_data1 = []
train_gts1 = []

for smp in train_dataset1.as_numpy_iterator():
    x, y = smp
    if y != 1:
        train_data1.append(np.squeeze(lmodel.predict(np.expand_dims(x, axis=0), verbose=0)))
        train_gts1.append(np.clip(y, 0, 1))
        
val_data1 = []
val_gts1 = []
for smp in test_dataset1.as_numpy_iterator():
    x, y = smp
    # if y != 1:
    val_data1.append(np.squeeze(lmodel.predict(np.expand_dims(x, axis=0), verbose=0)))
    val_gts1.append(np.clip(y, 0, 1))
        
def read_one_train(smpl):
    y_f = train_gts1[smpl]
    x_f = train_data1[smpl-INTV:smpl]
    
    return np.array(x_f).astype("float32"), y_f.astype("int32")
    
def lpreprocess_train(idx):
    spec, audio = tf.numpy_function(read_one_train, [idx], [tf.float32, tf.int32])
    
    return spec,  audio

def read_one_test(smpl):            
    y_f = val_gts1[smpl]
    x_f = val_data1[smpl-INTV:smpl]
    
    return np.array(x_f).astype("float32"), y_f.astype("int32")
    
def lpreprocess_test(idx):
    spec, audio = tf.numpy_function(read_one_test, [idx], [tf.float32, tf.int32])

    return spec,  audio


t_samples1 = np.arange(INTV, int(len(train_data1)))
v_samples1 = np.arange(INTV, len(val_data1))

ltrain_dataset1 = tf.data.Dataset.from_tensor_slices((t_samples1,))
ltrain_dataset1 = ltrain_dataset1.map(lpreprocess_train, num_parallel_calls=tf.data.AUTOTUNE)

ltest_dataset1 = tf.data.Dataset.from_tensor_slices((v_samples1,))
ltest_dataset1 = ltest_dataset1.map(lpreprocess_test, num_parallel_calls=tf.data.AUTOTUNE)

tmp_model.load_weights('./ckpt/'+tmp_model.name+'_'+RUN_NAME+'_best.our_data' + os.getenv('USER') + '.h5')
ev_res = tmp_model.evaluate(ltest_dataset1.batch(BATCH_SIZE))


preds1 = tmp_model.predict(ltest_dataset1.batch(BATCH_SIZE))
second_model_pretrain_metrics_on_val = precision_recall_fscore_support(val_gts1[INTV:], preds1 > 0.5, average=None)
weight_decay = 0.00001
LR_RATE = 0.0001
BATCH_SIZE = 32
optimizer = tfa.optimizers.AdamW(learning_rate=LR_RATE, weight_decay=weight_decay)

#optimizer = tf.optimizers.Adam(learning_rate=LR_RATE)

lfn = tf.keras.losses.BinaryCrossentropy()

msca =  [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

checkpoint_filepath = './ckpt/'+tmp_model.name+'_'+RUN_NAME+'_ftune25_best.our_data' + os.getenv('USER') + '.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


tmp_model.compile(optimizer=optimizer, loss= lfn, metrics=msca)

hist = tmp_model.fit(ltrain_dataset1.shuffle(200).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), 
          validation_data = ltest_dataset1.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), 
          epochs=25,
          callbacks=[model_checkpoint_callback]
         )

tmp_model.load_weights('./ckpt/'+tmp_model.name+'_'+RUN_NAME+'_ftune25_best.our_data' + os.getenv('USER') + '.h5')
preds100 = tmp_model.predict(ltest_dataset1.batch(BATCH_SIZE))

ev = tmp_model.evaluate((ltest_dataset1.batch(BATCH_SIZE)))

second_model_finetune_metrics_on_val = precision_recall_fscore_support(val_gts1[INTV:], preds100 > 0.5, average=None)
second_model_finetune_metrics_on_val2 = balanced_accuracy_score(val_gts1[INTV:], preds100 > 0.5)

all_results = {
    'second_model_pretrain_metrics': [i.tolist() for i in second_model_pretrain_metrics],
    'second_model_pretrain_metrics_on_val': [i.tolist() for i in second_model_pretrain_metrics_on_val],
    'second_model_finetune_metrics_on_val': [i.tolist() for i in second_model_finetune_metrics_on_val],
    'second_model_finetune_metrics_on_val_acc': float(second_model_finetune_metrics_on_val2),
}

import json
with open('../results/results_{}.inner.our_data.demo_task2.json'.format(test_subj[0]), 'w') as f:
    json.dump(all_results, f)