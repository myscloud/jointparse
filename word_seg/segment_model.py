import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from math import sqrt

from word_seg.custom_heap import CustomHeap

# parameters
learning_rate = 0.01
beam_size = 10
margin_loss_discount = 0.2
dropout_rate = 0.2
l2_lambda = 10e-4

subword_lstm_dim = 100
bigram_lstm_dim = 100
sent_lstm_dim = 200
hidden_dim = 150

subword_vocab_size = 100004
bigram_vocab_size = 60436
embedding_dim = 64

n_kept_model = 5
n_class = 4


def nn_bilstm_input(input_data, scope_name, lstm_dim):
    with tf.variable_scope(scope_name) as vs:
        fw_cell = rnn.BasicLSTMCell(lstm_dim, reuse=tf.get_variable_scope().reuse)
        bw_cell = rnn.BasicLSTMCell(lstm_dim, reuse=tf.get_variable_scope().reuse)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_data, dtype=tf.float32, scope=vs)

    (fw_outputs, bw_outputs) = outputs
    bilstm_output = tf.concat([fw_outputs[:, -1, :], bw_outputs[:, -1, :]], axis=1)
    return bilstm_output


def nn_input_layer(subwords, bigrams, subword_emb, bigram_emb):
    mapped_subwords = tf.nn.embedding_lookup(subword_emb, subwords)
    mapped_bigrams = tf.nn.embedding_lookup(bigram_emb, bigrams)
    subword_output = nn_bilstm_input(mapped_subwords, 'subword_lstm', subword_lstm_dim)
    bigram_output = nn_bilstm_input(mapped_bigrams, 'bigram_lstm', bigram_lstm_dim)
    concat_output = tf.concat([subword_output, bigram_output], axis=1)
    return concat_output


def nn_lstm_sentence_layer(input_vec):
    lstm_input = tf.expand_dims(input_vec, axis=0)

    with tf.variable_scope('sentence_lstm') as vs:
        fw_cell = rnn.BasicLSTMCell(sent_lstm_dim)
        bw_cell = rnn.BasicLSTMCell(sent_lstm_dim)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, lstm_input, dtype=tf.float32, scope=vs)

    (fw_outputs, bw_outputs) = outputs
    lstm_outputs = tf.concat([fw_outputs, bw_outputs], axis=2)
    return tf.squeeze(lstm_outputs, axis=0)


def nn_hidden_layer(input_vec):
    input_dim = (sent_lstm_dim * 2)
    hidden_weights = tf.Variable(tf.truncated_normal([input_dim, hidden_dim], stddev=1.0/sqrt(hidden_dim)),
                                 name='weights/hidden')
    hidden_bias = tf.Variable(tf.zeros([hidden_dim]), name='bias/hidden')
    return tf.nn.tanh(tf.matmul(input_vec, hidden_weights) + hidden_bias)


def nn_output_layer(hidden_vec):
    dropped_hidden_vec = tf.nn.dropout(hidden_vec, dropout_rate)
    output_weights = tf.Variable(tf.truncated_normal([hidden_dim, n_class], stddev=1.0/sqrt(n_class)),
                                 name='weights/output')
    output_bias = tf.Variable(tf.zeros([n_class]), name='bias/output')
    output = tf.matmul(hidden_vec, output_weights) + output_bias
    dropped_output = tf.matmul(dropped_hidden_vec, output_weights) + output_bias
    normalized_output = tf.nn.softmax(output)
    return normalized_output, dropped_output


def nn_calc_loss(predicted_output, labels):
    mapped_labels = tf.one_hot(labels, n_class, on_value=1.0, off_value=0.0, axis=-1)
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_output, labels=mapped_labels))
    all_weights = [tensor for tensor in tf.global_variables() if 'weights' in tensor.name]
    l2_score = l2_lambda * sum([tf.nn.l2_loss(tensor) for tensor in all_weights])
    all_loss = ce_loss + l2_score
    return all_loss

input_subwords = tf.placeholder(tf.int32, [None, None], name='placeholder/subwords')
input_bigrams = tf.placeholder(tf.int32, [None, None], name='placeholder/bigrams')
labels_index = tf.placeholder(tf.int32, [None], name='placeholder/labels')

subword_embedding = tf.Variable(tf.zeros([subword_vocab_size, embedding_dim]), trainable=False, name='subword_emb')
bigram_embedding = tf.Variable(tf.zeros([bigram_vocab_size, embedding_dim]), name='bigram_emb')

processed_input_vec = nn_input_layer(input_subwords, input_bigrams, subword_embedding, bigram_embedding)
sent_input_vec = nn_lstm_sentence_layer(processed_input_vec)
hidden_output = nn_hidden_layer(sent_input_vec)
normalized_output_vec, trained_output = nn_output_layer(hidden_output)

loss = nn_calc_loss(trained_output, labels_index)
optimize = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=n_kept_model)


class SegmentModel:
    def __init__(self, transition_prob, model_path=None, parameters=None):
        self.session = tf.Session()
        self.trans_prob = transition_prob
        if model_path is not None:
            saver.restore(self.session, model_path)
        else:
            self.session.run(init)
            self.assign_parameters(parameters)

    def assign_parameters(self, parameters):
        subword_placeholder = tf.placeholder(tf.float32, [subword_vocab_size, embedding_dim], name='subword_emb_pl')
        bigram_placeholder = tf.placeholder(tf.float32, [bigram_vocab_size, embedding_dim], name='bigram_emb_pl')
        assign_subword = tf.assign(subword_embedding, subword_placeholder)
        assign_bigram = tf.assign(bigram_embedding, bigram_placeholder)

        feed_dict = {
            subword_placeholder: parameters['subword_embedding'],
            bigram_placeholder: parameters['bigram_embedding']
        }
        self.session.run([assign_subword, assign_bigram], feed_dict=feed_dict)

    def train(self, subwords, bigrams, labels):
        feed_dict = SegmentModel.get_feed_dict(subwords, bigrams, labels)
        _, sent_loss = self.session.run([optimize, loss], feed_dict=feed_dict)
        return sent_loss

    def predict(self, subwords, bigrams, no_output):
        feed_dict = SegmentModel.get_feed_dict(subwords, bigrams)
        predicted_scores = self.session.run(normalized_output_vec, feed_dict=feed_dict)

        decoded_sequences = decode(predicted_scores, self.trans_prob, no_output)
        sequences_only = [x[1] for x in decoded_sequences]
        return sequences_only

    def save_model(self, save_path, global_step):
        saver.save(self.session, save_path, global_step=global_step)

    @staticmethod
    def get_feed_dict(subwords, bigrams, labels=None):
        feed_dict = {
            input_subwords: subwords,
            input_bigrams: bigrams
        }
        if labels is not None:
            feed_dict[labels_index] = labels
        return feed_dict

# ------------------------------------------------------------------------------------
# loss function
# ------------------------------------------------------------------------------------


def calc_sent_loss(gold_labels, predicted_scores, trans_prob):
    decoded_outputs = decode(predicted_scores, trans_prob, 1)
    (predicted_score, predicted_labels) = decoded_outputs[0]
    gold_score = calc_gold_label_score(gold_labels, predicted_scores, trans_prob)
    margin_loss = calc_margin_loss(gold_labels, predicted_labels)
    sent_loss = predicted_score + margin_loss - gold_score
    return max(0, sent_loss)


def decode(predicted_scores, trans_prob, max_ans):
    beam_queue = CustomHeap(max_size=beam_size)

    # initialize
    for tag_idx in range(n_class):
        score = predicted_scores[0][tag_idx]
        beam_queue.add((score, [tag_idx]))

    for char_count in range(1, len(predicted_scores)):
        prev_items = beam_queue.get_items_with_score()
        beam_queue = CustomHeap(max_size=beam_size)

        for (prev_score, item) in prev_items:
            last_tag = item[-1]
            for new_tag in range(n_class):
                new_score = prev_score + (trans_prob[last_tag][new_tag] + predicted_scores[char_count][new_tag])
                beam_queue.add((new_score, item + [new_tag]))

    sorted_answer = beam_queue.get_items_with_score()
    return sorted_answer[:max_ans]


def calc_gold_label_score(gold_label, predicted_scores, trans_prob):
    score = predicted_scores[0][gold_label[0]]
    for tag_idx in range(1, len(gold_label)):
        last_tag = gold_label[tag_idx-1]
        curr_tag = gold_label[tag_idx]
        score += (trans_prob[last_tag][curr_tag] + predicted_scores[tag_idx][curr_tag])

    return score


def calc_margin_loss(gold_labels, predicted_labels):
    correct_count = 0
    for predict, gold in zip(predicted_labels, gold_labels):
        correct_count += 1 if (predict == gold) else 0

    return margin_loss_discount * correct_count


