import tensorflow as tf
import numpy as np
import json
import time
import os
import sqlite3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Vocabulary import Vocabulary
from MainModel import Encoder, Decoder
from evaluate import evaluate


def reinitialize_vocab(v, hparams, credentials, offset, verbose=False):
    #tf.keras.backend.clear_session()
    v.remove_inputs()
    v.create_inputs_from_indexed(credentials, offset=offset, limit_main=hparams['NUM_EXAMPLES'], verbose=verbose)
    dataset = create_dataset(v, hparams['BATCH_SIZE'], hparams['NUM_EXAMPLES'])
    if verbose: print('Vocabulary reinitialized!')
    return v, dataset

def initialize_model_from_local(path, hparams, de_tokenize=False, verbose=False):
    start = time.time()

    conn = sqlite3.connect(path)
    c = conn.cursor()
    v = Vocabulary(max_len=hparams['MAX_LEN'])
    v.load_vocab_from_local(c, hparams['VOCAB'], verbose)
    c.close()
    conn.close()

    if de_tokenize: v.de_tokenize_data()
    if verbose: print('Vocabulary created!')

    enc, dec, opt = create_model(hparams)

    print('Time to initialize model {:.2f} min | {:.2f} hrs\n'.format((time.time()-start)/60, (time.time()-start)/3600))
    return v, enc, dec, opt

def load_hyper_params(path):
    with open(path, 'r') as f: data = json.load(f)

    for k in data:
        if data[k] == 'None':
            data[k] = None
    return data

def create_model(hparams):
    enc = Encoder(hparams['VOCAB'], hparams['BATCH_SIZE'], hparams['EMBEDDING'],
                  hparams['RNN1'], hparams['RNN2'], hparams['RNN_TYPE'], hparams['BIDIRECTIONAL'], hparams['MERGE_MODE'], hparams['DROPOUT_ENC'])
    dec = Decoder(hparams['VOCAB'], hparams['BATCH_SIZE'], hparams['EMBEDDING'],
                  hparams['RNN1'], hparams['RNN2'], hparams['RNN_TYPE'], hparams['DROPOUT_DEC'])
    #opt = tf.keras.optimizers.Adam(learning_rate=hparams['LR'])
    opt = tf.keras.optimizers.SGD(learning_rate=hparams['LR'], momentum=0.3)
    return enc, dec, opt

def create_dataset(v, batch_size, buffer_size):
    v.tokenize_data()
    dataset = tf.data.Dataset.from_tensor_slices((np.array(v.inp, dtype=np.int32), np.array(v.tar, dtype=np.int32)))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def save_plot(path, plt_loss):
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(path + 'plot.txt', 'a', encoding='utf-8') as f:
        f.write(str(plt_loss[-1].numpy()) + '\n')


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    #fig.suptitle('Attention weights')
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

#------------------------ Legacy ---------------------------
def initialize_model(v, hparams, credentials, offset=0, from_indexed=True, create_ds=True, de_tokenize=False, verbose=False):
    start = time.time()
    if not v.vocab_load: v.load_bigquery_vocab_from_indexed(credentials, hparams['VOCAB_DB'], hparams['VOCAB'], verbose)

    if from_indexed: v.create_inputs_from_indexed(credentials,
                                                  offset=offset,
                                                  limit_main=hparams['NUM_EXAMPLES'],
                                                  verbose=True)  # <--- False
    else: v.create_inputs(credentials,
                          offset=offset,
                          limit_main=hparams['NUM_EXAMPLES'],
                          verbose=True)  # <--- False

    if verbose: print('Vocabulary created!')

    if create_ds: dataset = create_dataset(v, hparams['BATCH_SIZE'], hparams['NUM_EXAMPLES'])
    enc, dec, opt = create_model(hparams)

    if de_tokenize: v.de_tokenize_data()
    print('Time to initialize model {:.2f} min | {:.2f} hrs\n'.format((time.time()-start)/60, (time.time()-start)/3600))

    if create_ds: return v, dataset, enc, dec, opt
    return v, enc, dec, opt
