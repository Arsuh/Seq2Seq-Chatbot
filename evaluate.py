import tensorflow as tf
import numpy as np

def evaluate(text, v, enc, dec, max_len):
    inp = np.array(v.preproc(text), dtype=np.float32)
    inp = tf.convert_to_tensor(inp)
    inp = tf.expand_dims(inp, axis=0)

    attention_plot = np.zeros((max_len, max_len))

    result = ''
    h1, h2 = enc.initialize_hidden(batch=1)

    enc_out, h1, h2 = enc.call(inp, h1, h2, training=False)
    dec_inp = tf.expand_dims([1], axis=0)  # SOS
    result += '<SOS> '
    for i in range(v.max_len):
        pred, h1, h2, attention_weights = dec.call(dec_inp, enc_out, h1, h2, training=False)

        pred = tf.nn.softmax(pred, axis=1)
        pred_id = tf.argmax(pred[0]).numpy()
        #pred_id = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[i] = attention_weights.numpy()

        result += v.idx2word[pred_id] + ' '
        if pred_id == 2:  # EOS
            return result, text, attention_plot

        dec_inp = tf.expand_dims([pred_id], axis=0)

    return result[:-1], text, attention_plot
