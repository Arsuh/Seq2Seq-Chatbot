import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import shutil
import random

from google.oauth2 import service_account

from evaluate import evaluate
from MainModel import loss_fnc
from helper import initialize_model, load_hyper_params, reinitialize_vocab, create_dataset, create_model
from Vocabulary import Vocabulary

ckpt_path = './checkpoints'
#enc_prefix = os.path.join(ckpt_path+'/enc', 'ckpt-{epoch}')
#dec_prefix = os.path.join(ckpt_path+'/dec', 'ckpt-{epoch}')
ckpt_prefix = os.path.join(ckpt_path, 'ckpt')


def train_step(hparams, inp, tar, enc_h1, enc_h2):
    global enc, dec, opt
    loss = 0
    with tf.GradientTape() as tape:
        enc_out, enc_h1, enc_h2 = enc(inp, enc_h1, enc_h2)
        dec_h1, dec_h2 = enc_h1, enc_h2
        dec_inp = tf.expand_dims([1]*hparams['BATCH_SIZE'], 1)

        for t in range(1, tar.shape[1]):
            pred, dec_h1, dec_h2, _ = dec(dec_inp, enc_out, dec_h1, dec_h2)

            loss += loss_fnc(tar[:, t], pred)
            dec_inp = tf.expand_dims(tar[:, t], 1)

    batch_loss = (loss/int(tar.shape[1]))
    variables = enc.trainable_variables + dec.trainable_variables
    gradients = tape.gradient(loss, variables)
    opt.apply_gradients(zip(gradients, variables))

    return batch_loss



test_sentences = ['Hello!',
                  'How are you?',
                  'Tomorow is my birthday, but I keep feeling sad...',
                  'What is your name sir?',
                  'Artificial intelligence will take over the world some day!',
                  'Can you please bring me some water?',
                  'Come on! This is the easiest thing you are supposed to do!',
                  'My name is Thomas!']

def train(hparams, credentials, offset=0, saving=True, verbose=True):
    global enc, dec, opt
    start = time.time()
    v = Vocabulary(max_len=hparams['MAX_LEN'])
    v.load_bigquery_vocab_from_indexed(credentials, hparams['VOCAB_DB'], hparams['VOCAB'], verbose)
    v.create_inputs_from_indexed(credentials,
                                 offset=offset,
                                 limit_main=hparams['NUM_EXAMPLES'],
                                 verbose=True)  # <--- False

    if verbose: print('Vocabulary created!')
    dataset = create_dataset(v, hparams['BATCH_SIZE'], hparams['NUM_EXAMPLES'])
    if verbose: print('Time to initialize model {:.2f} min | {:.2f} hrs\n'.format((time.time()-start)/60, (time.time()-start)/3600))
    del start

    if hparams['NUM_EXAMPLES'] == None:
        N_BATCH = hparams['MAX_EXAMPLES'] // hparams['BATCH_SIZE']
    else:
        N_BATCH = hparams['NUM_EXAMPLES'] // hparams['BATCH_SIZE']

    if saving: checkpoint = tf.train.Checkpoint(optimizer=opt, encoder=enc, decoder=dec)

    plt_loss = []
    for epoch in range(1, hparams['EPOCHS']+1):
        h1, h2 = enc.initialize_hidden()

        total_loss = 0
        for (batch, (inp, tar)) in enumerate(dataset.take(N_BATCH)):
            batch_time = time.time()
            batch_loss = train_step(hparams, inp, tar, h1, h2)
            total_loss += batch_loss

            print('  >>> Epoch: {} | Batch: {}\\{} | Loss: {:.4f} | Time: {:.2f} sec'
                  .format(epoch, batch+1, N_BATCH, batch_loss, time.time() - batch_time))

        print('Epoch: {} | Loss: {:.4f}'.format(epoch+1, total_loss/N_BATCH))
        plt_loss.append(total_loss/N_BATCH)

        sentences = random.choices(test_sentences, k=2)
        result1, text1, _ = evaluate(sentences[0], v, enc, dec, hparams['MAX_LEN'])
        result2, text2, _ = evaluate(sentences[1], v, enc, dec, hparams['MAX_LEN'])
        print(50*'+')
        print(text1)
        print(result1)
        print(text2)
        print(result2)
        print(50*'+')

        if saving:
            print('Saving model...')
            checkpoint.save(file_prefix=ckpt_prefix)

    return plt_loss

def multi_initializer_train(hparams, credentials, saving=True, verbose=True):
    global enc, dec, opt
    v = Vocabulary(max_len=hparams['MAX_LEN'])
    v.load_bigquery_vocab_from_indexed(credentials, hparams['VOCAB_DB'], hparams['VOCAB'], verbose)
    if verbose: print('Vocabulary created!')

    if hparams['NUM_EXAMPLES'] == None:
        N_BATCH = hparams['MAX_EXAMPLES'] // hparams['BATCH_SIZE']
    else:
        N_BATCH = hparams['NUM_EXAMPLES'] // hparams['BATCH_SIZE']

    if saving: checkpoint = tf.train.Checkpoint(optimizer=opt, encoder=enc, decoder=dec)

    plt_loss = []
    for epoch in range(1, hparams['EPOCHS']+1):
        h1, h2 = enc.initialize_hidden()

        reps = int(hparams['MAX_EXAMPLES']//hparams['NUM_EXAMPLES']) if hparams['OFFSET_REP']=='max' else int(hparams['OFFSET_REP'])
        offset = 0
        total_loss = 0
        for rep in range(reps):
            v, dataset = reinitialize_vocab(v, hparams, credentials, offset, verbose=True)

            for (batch, (inp, tar)) in enumerate(dataset.take(N_BATCH)):
                batch_time = time.time()
                batch_loss = train_step(hparams, inp, tar, h1, h2)
                total_loss += batch_loss

                print('  >>> Epoch: {} | Batch: {}\\{} | Loss: {:.4f} | Time: {:.2f} sec'
                    .format(epoch, batch+1, N_BATCH, batch_loss, time.time() - batch_time))
            
            offset += hparams['NUM_EXAMPLES']
            tf.keras.backend.clear_session()
            print(' >>> Rep: {} done!'.format(rep + 1))

        print('Epoch: {} | Loss: {:.4f}'.format(epoch, total_loss/(N_BATCH*reps)))
        plt_loss.append(total_loss/N_BATCH)

        sentences = random.choices(test_sentences, k=2)
        result1, text1, _ = evaluate(sentences[0], v, enc, dec, hparams['MAX_LEN'])
        result2, text2, _ = evaluate(sentences[1], v, enc, dec, hparams['MAX_LEN'])
        print(50*'+')
        print(text1)
        print(result1)
        print(text2)
        print(result2)
        print(50*'+')

        if saving:
            print('Saving model...')
            checkpoint.save(file_prefix=ckpt_prefix)
    return plt_loss


if __name__ == '__main__':
    if os.path.isdir('./__pycache__/'): shutil.rmtree(path='./__pycache__/', ignore_errors=True, onerror=None)

    hparams = load_hyper_params('./hyper_parameters_test.json')
    credentials = service_account.Credentials.from_service_account_file(hparams['CREDENTIALS_PATH'])
    enc, dec, opt = create_model(hparams)

    if hparams['TRAINING_MODE'] == 'single': plt_loss = train(hparams, credentials, saving=False)
    elif hparams['TRAINING_MODE'] == 'multi': plt_loss = multi_initializer_train(hparams, credentials, saving=False)
    else: raise Exception('Please enter a valid TRAINING_MODE: \'single\' or \'multi\' '
                          '(for \'multi\' please use OFFSET_REP >= 2 or \'max\')')  
    
    plt.plot(plt_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
