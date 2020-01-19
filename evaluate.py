import tensorflow as tf
import numpy as np

from helper import load_hyper_params, initialize_model
from Vocabulary import Vocabulary


def evaluate(text, v, enc, dec):
    inp = np.array(v.preproc(text), dtype=np.float32)
    inp = tf.convert_to_tensor(inp)
    inp = tf.expand_dims(inp, axis=0)

    result = ''
    h1, h2 = enc.initialize_hidden(batch=1)

    enc_out, h1, h2 = enc.call(inp, h1, h2)
    dec_inp = tf.expand_dims([1], axis=0)  # SOS
    result += '<SOS> '
    for _ in range(v.max_len):
        pred, h1, h2, attention_wieghts = dec.call(dec_inp, enc_out, h1, h2)

        pred = tf.nn.softmax(pred, axis=1)
        pred_id = tf.argmax(pred[0]).numpy()
        #pred_id = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()

        result += v.idx2word[pred_id] + ' '
        if pred_id == 2:  # EOS
            return result, text, attention_wieghts

        dec_inp = tf.expand_dims([pred_id], axis=0)

    return result[:-1], text, attention_wieghts
