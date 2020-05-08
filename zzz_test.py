import tensorflow as tf

from google.oauth2 import service_account
from google.cloud import bigquery

import matplotlib.pyplot as plt
import numpy as np
import json
import re
import os
import time
import random
import shutil

from Vocabulary import Vocabulary
from MainModel import loss_fnc
from helper import *
from evaluate import evaluate
from train import *


hparams_path = './checkpoints/hparams.json'
if __name__ == '__main__':
    if os.path.isdir('./__pycache__/'): shutil.rmtree(path='./__pycache__/', ignore_errors=True, onerror=None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    hparams = load_hyper_params(hparams_path)
    credentials = service_account.Credentials.from_service_account_file(hparams['CREDENTIALS_PATH'])
    enc, dec, opt = create_model(hparams)

    v = Vocabulary(max_len=hparams['MAX_LEN'])
    v.load_bigquery_vocab_from_indexed(credentials, hparams['VOCAB_DB'], hparams['VOCAB'], True)
    print('Vocab')

    checkpoint = tf.train.Checkpoint(optimizer=opt, encoder=enc, decoder=dec)
    checkpoint.restore('./checkpoints/checkpoints-125ep-3/ckpt-5')
    print('checkpoint')

    result, _, _ = evaluate(u'The world is changing once again!', v, enc, dec, hparams['MAX_LEN'])
    print(result)