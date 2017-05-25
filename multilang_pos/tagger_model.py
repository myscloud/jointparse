import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from math import sqrt

# network parameters
learning_rate = 0.1
gaussian_noise = 0.2
dropout_prob = 0.3

word_vocab_size = 100004
subword_vocab_size = 100004
word_embedding_dim = 128
subword_embedding_dim = 100

input_subword_dim = 100
input_dim = 100
hidden_dim = 100

output_types = ['tag', 'word_freq']
n_output = dict()
n_output['tag'] = 15
n_output['word_freq'] = 14

# other parameters
n_saved_models = 2

x = dict()
x['words'] = tf.placeholder(tf.int32, [None, None], name='input_words')
x['subwords'] = tf.placeholder(tf.int32, [None, None], name='input_subwords')
x['subwords_len'] = tf.placeholder(tf.int32, [None], name='input_subwords_len')
label = dict()
label['tag'] = tf.placeholder(tf.int32, [None], name='label_tags')
label['word_freq'] = tf.placeholder(tf.int32, [None], name='label_word_freq')


def nn_run_subword_processing(subwords):
    with tf.variable_scope('subword_lstm'):
        subword_embedding = tf.Variable(tf.random_uniform([subword_vocab_size, subword_embedding_dim],
                                                          minval=-0.1, maxval=0.1), name='weights/subword_embedding')
        mapped_subwords = tf.nn.embedding_lookup(subword_embedding, subwords)
        lstm_fw_cell = rnn.BasicLSTMCell(input_subword_dim)
        lstm_bw_cell = rnn.BasicLSTMCell(input_subword_dim)
        lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, mapped_subwords, dtype=tf.float32)
        (fw_outputs, bw_outputs) = lstm_outputs

        return tf.concat([fw_outputs[:, -1, :], bw_outputs[:, -1, :]], 1)


def nn_run_input_layer(input_dict):
    word_embedding = tf.Variable(tf.random_uniform([word_vocab_size, word_embedding_dim],
                                                   minval=-0.1, maxval=0.1), name='weights/word_embedding')
    mapped_words = tf.nn.embedding_lookup(word_embedding, input_dict['words'])
    subword_vec = nn_run_subword_processing(input_dict['subwords'])
    expanded_subword_vec = tf.expand_dims(subword_vec, 0)
    word_subword = tf.concat([mapped_words, expanded_subword_vec], axis=2)

    with tf.variable_scope('word_lstm'):
        lstm_fw_cell = rnn.BasicLSTMCell(input_dim)
        lstm_bw_cell = rnn.BasicLSTMCell(input_dim)
        lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, word_subword, dtype=tf.float32)

    concat_data = tf.concat([lstm_outputs[0], lstm_outputs[1]], axis=2)
    input_data = tf.reshape(concat_data, [-1, input_dim*2])
    input_noise = tf.Variable(tf.truncated_normal([1, input_dim*2], stddev=gaussian_noise), name='input_noise')

    return tf.add(input_data, input_noise)


def nn_run_hidden_layer(input_vec):
    hidden_weight = tf.Variable(tf.truncated_normal([input_dim*2, hidden_dim], stddev=1.0/sqrt(hidden_dim)),
                                name='weights/hidden_layer')
    hidden_bias = tf.Variable(tf.zeros([hidden_dim]), name='bias/hidden_layer')
    hidden_output = tf.nn.relu(tf.add(tf.matmul(input_vec, hidden_weight), hidden_bias))
    return hidden_output


def nn_classify(hidden_vec):
    out_weights = dict()
    out_biases = dict()
    outputs = dict()

    for output_type in output_types:
        out_weights[output_type] = tf.Variable(tf.truncated_normal([hidden_dim, n_output[output_type]],
                                                                   stddev=1.0/sqrt(n_output[output_type])),
                                               name='weights/out_'+output_type)
        out_biases[output_type] = tf.Variable(tf.zeros([n_output[output_type]]), name='bias/out_'+output_type)
        outputs[output_type] = tf.matmul(hidden_vec, out_weights[output_type]) + out_biases[output_type]

    return outputs


def nn_calculate_loss(predicted_outputs):
    all_loss = 0.0
    for output_type in output_types:
        y = tf.one_hot(label[output_type], n_output[output_type], on_value=1.0, off_value=0.0, axis=-1)
        dropped_output = tf.nn.dropout(predicted_outputs[output_type], dropout_prob)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=dropped_output))
        all_loss += ce_loss

    return all_loss


processed_input = nn_run_input_layer(x)
h = nn_run_hidden_layer(processed_input)
predicted_outputs = nn_classify(h)
loss = nn_calculate_loss(predicted_outputs)
optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=n_saved_models)


class TaggerModel:
    def __init__(self, model_path=None):
        self.session = tf.Session()
        if model_path is not None:
            saver.restore(self.session, model_path)
        else:
            self.session.run(init)

    def train(self, input_list, labels):
        feed_dict = TaggerModel.get_feed_dict(input_list, labels)
        _, iter_loss = self.session.run([optimize, loss], feed_dict=feed_dict)
        return iter_loss

    def predict(self, input_list):
        feed_dict = TaggerModel.get_feed_dict(input_list)
        prediction = self.session.run(predicted_outputs, feed_dict=feed_dict)
        return prediction['tag']

    def save_model(self, save_path, global_step):
        saver.save(self.session,save_path, global_step=global_step)

    @staticmethod
    def get_feed_dict(input_list, labels=None):
        feed_dict = dict()
        feed_dict[x['words']] = input_list[0]
        feed_dict[x['subwords']] = input_list[1]
        feed_dict[x['subwords_len']] = input_list[2]

        if labels is not None:
            feed_dict[label['tag']], feed_dict[label['word_freq']] = labels

        return feed_dict

