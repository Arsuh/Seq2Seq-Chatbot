from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

EMB = 256
RNN1 = 512
RNN2 = 512
RNN_TYPE = 'lstm'
DROPOUT = 0.2


class Encoder(tf.keras.Model):
    def __init__(self, vocab_sz, batch_sz,
                 _emb=EMB, _rnn1=RNN1, _rnn2=RNN2, _rnn_type=RNN_TYPE, bidirectional=True, _merge_mode='ave', _dr=DROPOUT):
        super(Encoder, self).__init__()

        self.vocab_sz = vocab_sz + 4
        self.batch_sz = batch_sz
        self.construct_model(_emb, _rnn1, _rnn2, _rnn_type, bidirectional, _merge_mode, _dr)
        # self.build(tf.TensorShape([self.batch_sz, None]))

    def construct_model(self, _emb, _rnn1, _rnn2, _rnn_type, bidirectional, _merge_mode, _dr):
        self.emb_dim = _emb
        self.rnn1_units = _rnn1
        self.rnn2_units = _rnn2
        self.rnn_type = _rnn_type
        self.bidirectional = bidirectional
        if self.bidirectional == False:
            self._merge_mode = None
        else:
            self._merge_mode = _merge_mode
        self.dropout = _dr

        self.embedding = tf.keras.layers.Embedding(self.vocab_sz, self.emb_dim,
                                                   batch_input_shape=[self.batch_sz, None],)

        if self.bidirectional:
            if self.rnn_type == 'gru':
                rnn1_forward = gru_fnc(self.rnn1_units, self.dropout)
                rnn1_backward = gru_fnc(
                    self.rnn1_units, self.dropout, inverse=True)
                rnn2_forward = gru_fnc(self.rnn2_units, self.dropout)
                rnn2_backward = gru_fnc(
                    self.rnn2_units, self.dropout, inverse=True)
            elif self.rnn_type == 'lstm':
                rnn1_forward = lstm_fnc(self.rnn1_units, self.dropout)
                rnn1_backward = lstm_fnc(
                    self.rnn1_units, self.dropout, inverse=True)
                rnn2_forward = lstm_fnc(self.rnn2_units, self.dropout)
                rnn2_backward = lstm_fnc(
                    self.rnn2_units, self.dropout, inverse=True)
            else:
                raise Exception(
                    'RNN TYPE not recognized! Please use \'gru\' or \'lstm\'!')

            self.rnn1 = tf.keras.layers.Bidirectional(
                rnn1_forward, backward_layer=rnn1_backward, merge_mode=self._merge_mode)
            self.rnn2 = tf.keras.layers.Bidirectional(
                rnn2_forward, backward_layer=rnn2_backward, merge_mode=self._merge_mode)
        else:
            if self.rnn_type == 'gru':
                self.rnn1 = gru_fnc(self.rnn1_units, self.dropout)
                self.rnn2 = gru_fnc(self.rnn2_units, self.dropout)
            elif self.rnn_type == 'lstm':
                self.rnn1 = lstm_fnc(self.rnn1_units, self.dropout)
                self.rnn2 = lstm_fnc(self.rnn2_units, self.dropout)
            else:
                raise Exception(
                    'RNN TYPE not recognized! Please use \'gru\' or \'lstm\'!')

        self.W = tf.keras.layers.Dense(self.rnn1_units)

    def call(self, x, h1, h2, training=True):
        x = self.embedding(x)

        if self.rnn_type == 'gru' and self.bidirectional == False:
            output1, state1 = self.rnn1(x, initial_state=h1, training=training)
            output2, state2 = self.rnn2(self.W(output1), initial_state=h2, training=training)
            return output2, state1, state2

        output1 = self.rnn1(x, initial_state=h1, training=training)
        output2 = self.rnn2(self.W(output1[0]), initial_state=h2, training=training)
        # result, h1, h2
        return output2[0], output1[1] + output1[2], output2[1] + output2[2]

    def initialize_hidden(self, batch=None):
        if batch is None:
            batch = self.batch_sz

        nr_gru, nr_lstm = 2, 4
        if self.bidirectional == False:
            nr_gru = nr_gru // 2
            nr_lstm = nr_lstm // 2

        if self.rnn_type == 'gru':
            h1 = [tf.zeros((batch, self.rnn1_units)) for _ in range(nr_gru)]
            h2 = [tf.zeros((batch, self.rnn2_units)) for _ in range(nr_gru)]
        elif self.rnn_type == 'lstm':
            h1 = [tf.zeros((batch, self.rnn1_units)) for _ in range(nr_lstm)]
            h2 = [tf.zeros((batch, self.rnn2_units)) for _ in range(nr_lstm)]
        return h1, h2


class Decoder(tf.keras.Model):
    def __init__(self, vocab_sz, batch_sz,
                 _emb=EMB, _rnn1=RNN1, _rnn2=RNN2, _rnn_type=RNN_TYPE, _dr=DROPOUT):
        super(Decoder, self).__init__()

        self.vocab_sz = vocab_sz + 4
        self.batch_sz = batch_sz

        self.construct_model(_emb, _rnn1, _rnn2, _rnn_type, _dr)
        # self.build(tf.TensorShape([self.batch_sz, None]))

    def construct_model(self, _emb, _rnn1, _rnn2, _rnn_type, _dr):
        self.emb_dim = _emb
        self.rnn1_units = _rnn1
        self.rnn2_units = _rnn2
        self.rnn_type = _rnn_type
        self.dropout = _dr

        self.embedding = tf.keras.layers.Embedding(self.vocab_sz, self.emb_dim,
                                                   batch_input_shape=[self.batch_sz, None])

        if self.rnn_type == 'gru':
            self.rnn1 = gru_fnc(self.rnn1_units, self.dropout)
            self.rnn2 = gru_fnc(self.rnn2_units, self.dropout)
        elif self.rnn_type == 'lstm':
            self.rnn1 = lstm_fnc(self.rnn1_units, self.dropout)
            self.rnn2 = lstm_fnc(self.rnn2_units, self.dropout)
        else:
            raise Exception(
                'RNN TYPE not recognized! Plase use \'gru\' or \'lstm\'!')

        self.W = tf.keras.layers.Dense(self.rnn1_units)
        self.attention = BahdanauAttention(self.batch_sz, self.rnn1_units, self.rnn2_units)
        self.fc = tf.keras.layers.Dense(self.vocab_sz)
        #self.final = tf.keras.layers.Dense(self.vocab_sz, activation='softmax')

    def call(self, x, enc_output, h1, h2, training=True):
        context_vector, attention_weights = self.attention(enc_output, h1, h2)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)

        output1 = self.rnn1(x, training=training)
        output2 = self.rnn2(self.W(output1[0]), training=training)
        state2 = output2[1]
        output = output2[0]
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)
        #x = self.final(x)
        # result, h1, h2, attention_wieghts
        return x, output1[1], state2, attention_weights


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, batch_sz, rnn1_units, rnn2_units):
        super(BahdanauAttention, self).__init__()
        self.W_enc = tf.keras.layers.Dense(rnn2_units)  # <--- Encoder output
        self.W1 = tf.keras.layers.Dense(rnn1_units)  # <--- Hidden state 1
        self.W2 = tf.keras.layers.Dense(rnn2_units)  # <--- Hidden state 2
        self.W = tf.keras.layers.Dense(1)      # <--- Score

        # self.build(tf.TensorShape([batch_sz, None]))

    def call(self, output, h1, h2):
        h1_expand = tf.expand_dims(h1, axis=1)
        h2_expand = tf.expand_dims(h2, axis=1)

        x = self.W_enc(output)
        concat_rnns = tf.concat((self.W2(h2_expand), self.W1(h1_expand)), axis=2)
        x = BahdanauAttention.broadcast(x, concat_rnns.shape)

        score = self.W(tf.tanh(x + concat_rnns))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = output * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    @staticmethod
    def broadcast(tensor, shape):
        return tf.concat((tensor, tf.zeros([tensor.shape[0], tensor.shape[1], shape[2]-tensor.shape[2]])), axis=2)


def gru_fnc(units, dropout, inverse=False):
    return tf.keras.layers.GRU(units,
                               activation='tanh',
                               recurrent_activation='sigmoid',
                               recurrent_dropout=0.0,
                               unroll=False,
                               use_bias=True,
                               # reset_after=True,
                               return_sequences=True,
                               return_state=True,
                               stateful=False,
                               recurrent_initializer='glorot_uniform',
                               go_backwards=inverse,
                               dropout=dropout,
                               )


def lstm_fnc(units, dropout, inverse=False):
    return tf.keras.layers.LSTM(units,
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                recurrent_dropout=0.0,
                                unroll=False,
                                use_bias=True,
                                # reset_after=True,
                                return_sequences=True,
                                return_state=True,
                                stateful=False,
                                recurrent_initializer='glorot_uniform',
                                go_backwards=inverse,
                                dropout=dropout,
                                )


_loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_fnc(y_true, y_pred):
    loss = _loss(y_true, y_pred)
    mask = 1 - np.array_equal(y_true, 0)
    mask = tf.cast(mask, dtype=loss.dtype)
    return tf.reduce_mean(loss * mask)
