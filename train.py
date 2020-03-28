import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import random
from shutil import copyfile

from evaluate import evaluate
from MainModel import loss_fnc
from helper import initialize_model, load_hyper_params, save_plot
from Vocabulary import Vocabulary

ckpt_path = './checkpoints/'
ckpt_prefix = os.path.join(ckpt_path, 'ckpt')
hparams_path = './hyper_parameters_test.json'


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


def train(hparams, saving=True, plot_saving=True, verbose=True):
    global v, dataset, enc, dec, opt
    if hparams['NUM_EXAMPLES'] == None:
        N_BATCH = hparams['MAX_EXAMPLES'] // hparams['BATCH_SIZE']
    else:
        N_BATCH = hparams['NUM_EXAMPLES'] // hparams['BATCH_SIZE']

    if saving:
        checkpoint = tf.train.Checkpoint(optimizer=opt, encoder=enc, decoder=dec)
        copyfile(hparams_path, ckpt_path + '/hparams.json')

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
        if plot_saving:
            save_plot(ckpt_path, plt_loss)
    return plt_loss


if __name__ == '__main__':
    hparams = load_hyper_params(os.path.join(hparams_path))
    v, dataset, enc, dec, opt = initialize_model(
        hparams, from_indexed=True, create_ds=True, de_tokenize=False, verbose=True)

    plt_loss = train(hparams, saving=False, plot_saving=False)

    plt.plot(plt_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
