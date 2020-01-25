from google.oauth2 import service_account
import tensorflow as tf
import numpy as np
import json
import time
import os

from Vocabulary import Vocabulary
from MainModel import Encoder, Decoder


def initialize_model(hparams, from_indexed=True, create_ds=True, de_tokenize=True, verbose=False):
    start = time.time()
    if from_indexed:
        v = Vocabulary.create_inputs_from_indexed(service_account.Credentials.from_service_account_file(hparams['CREDENTIALS_PATH']),
                                                  max_len=hparams['MAX_LEN'],
                                                  vocab=hparams['VOCAB_DB'],
                                                  limit_main=hparams['NUM_EXAMPLES'],
                                                  limit_vocab=hparams['VOCAB'],
                                                  verbose=True)  # <--- False
    else:
        v = Vocabulary.create_inputs(service_account.Credentials.from_service_account_file(hparams['CREDENTIALS_PATH']),
                                     max_len=hparams['MAX_LEN'],
                                     vocab=hparams['VOCAB_DB'],
                                     limit_main=hparams['NUM_EXAMPLES'],
                                     limit_vocab=hparams['VOCAB'],
                                     verbose=True)  # <--- False

    if de_tokenize:
        v.de_tokenize_data()
    if verbose:
        print('Vocabulary created!')

    if create_ds:
        dataset = create_dataset(
            v, hparams['BATCH_SIZE'], hparams['NUM_EXAMPLES'])
    enc = Encoder(hparams['VOCAB'], hparams['BATCH_SIZE'], hparams['EMBEDDING'],
                  hparams['RNN1'], hparams['RNN2'], hparams['RNN_TYPE'], hparams['BIDIRECTIONAL'], hparams['MERGE_MODE'], hparams['DROPOUT_ENC'])
    dec = Decoder(hparams['VOCAB'], hparams['BATCH_SIZE'], hparams['EMBEDDING'],
                  hparams['RNN1'], hparams['RNN2'], hparams['RNN_TYPE'], hparams['DROPOUT_DEC'])
    opt = tf.keras.optimizers.Adam(learning_rate=hparams['LR'])
    #opt = tf.keras.optimizers.SGD(learning_rate=hparams['LR'], momentum=0.5)

    print('Time to initialize model {:.2f} min | {:.2f} hrs\n'.format(
        (time.time()-start)/60, (time.time()-start)/3600))

    if create_ds:
        return v, dataset, enc, dec, opt
    return v, enc, dec, opt


def load_hyper_params(path):
    with open(path, 'r') as f:
        data = json.load(f)

    for k in data:
        if data[k] == 'None':
            data[k] = None
    return data


def create_dataset(v, batch_size, buffer_size):
    v.tokenize_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        (np.array(v.inp, dtype=np.int32), np.array(v.tar, dtype=np.int32)))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def save_plot(path, plt_loss):
    if not os.path.isdir(path+'/'):
        os.mkdir(path+'/')
    with open(path + '/plot.txt', 'a', encoding='utf-8') as f:
        f.write(str(plt_loss[-1].numpy()) + '\n')
