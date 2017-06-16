import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from math import sqrt
from numpy import argmax

# parameters
dropout_prob = 0.5
n_kept_model = 2

k_word_candidate = 3
k_bpos_candidate = 3

# dimension
buffer_max_len = 300
stack_max_len = 200
actions_max_len = 400

lstm_dim = 100
subword_lstm_dim = 64
output_dim = 20

word_vocab_size = 100004
subword_vocab_size = 100004
word_emb_dim = 64
param_emb_dim = 30
candidate_emb_dim = 20

n_pos = 16
n_bpos = 61
n_action = 107
n_class = 106

# variables, placeholders, embeddings
buffer = tf.Variable(tf.zeros([1, buffer_max_len, lstm_dim]), trainable=False, name='input_buffer')
stack = tf.Variable(tf.zeros([1, stack_max_len, lstm_dim]), trainable=False, name='input_stack')
actions = tf.Variable(tf.zeros([1, actions_max_len, lstm_dim]), trainable=False, name='input_actions')

subword_ph = tf.placeholder(tf.int32, [None], name='placeholder_subword')
word_candidates_ph = tf.placeholder(tf.int32, [None, None], name='placeholder_word_candidates')
bpos_ph = tf.placeholder(tf.int32, [None, None], name='placeholder_bpos')
word_ph = tf.placeholder(tf.int32, None, name='placeholder_word')
subword_list_ph = tf.placeholder(tf.int32, [None], name='placeholder_subword_list')
pos_ph = tf.placeholder(tf.int32, None, name='placeholder_pos')
relation_action_ph = tf.placeholder(tf.int32, [None], name='placeholder_rel_action')
action_ph = tf.placeholder(tf.int32, None, name='placeholder_action')
output_mask = tf.placeholder(tf.float32, [None], name='placeholder_output_mask')
label = tf.placeholder(tf.int32, None, name='placeholder_output')

stack_len = tf.placeholder(tf.int32, name='placeholder_stack_len')
buffer_len = tf.placeholder(tf.int32, name='placeholder_buffer_len')
actions_len = tf.placeholder(tf.int32, name='placeholder_actions_len')
subwords_len = tf.placeholder(tf.int32, name='placeholder_subwords_len')

word_lm_emb = tf.Variable(tf.zeros([word_vocab_size, word_emb_dim]), trainable=False, name='embedding_word_lm')
subword_emb = tf.Variable(tf.zeros([subword_vocab_size, word_emb_dim]), trainable=False, name='embedding_subword')
word_emb_ph = tf.placeholder(tf.float32, [word_vocab_size, word_emb_dim])
subword_emb_ph = tf.placeholder(tf.float32, [word_vocab_size, word_emb_dim])
assign_word_embedding = tf.assign(word_lm_emb, word_emb_ph, validate_shape=False)
assign_subword_embedding = tf.assign(subword_emb, subword_emb_ph, validate_shape=False)

word_emb = tf.Variable(tf.random_uniform([word_vocab_size, candidate_emb_dim], minval=-0.1, maxval=0.1),
                       name='embedding_word')
bpos_emb = tf.Variable(tf.random_uniform([n_bpos, candidate_emb_dim], minval=-0.1, maxval=0.1), name='embedding_bpos')

pos_emb = tf.Variable(tf.random_uniform([n_pos, param_emb_dim], minval=-0.1, maxval=0.1), name='embedding_pos')
action_emb = tf.Variable(tf.random_uniform([n_action, param_emb_dim], minval=-0.1, maxval=0.1), name='embedding_action')


def nn_run_lstm_input(input_vec, dim, seq_len, scope_name):
    with tf.variable_scope(scope_name) as vs:
        lstm_fw_cell = rnn.BasicLSTMCell(dim, reuse=tf.get_variable_scope().reuse)

        lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_fw_cell, input_vec, dtype=tf.float32, scope=vs, sequence_length=seq_len)
        outputs = tf.reshape(lstm_outputs[:, 0, :], [1, dim])
        return outputs

with tf.name_scope('input_layer'):
    padded_buffer = tf.pad(buffer, [[0, 0], [0, buffer_max_len - buffer_len], [0, 0]])
    padded_stack = tf.pad(stack, [[0, 0], [0, stack_max_len - stack_len], [0, 0]])
    padded_actions = tf.pad(actions, [[0, 0], [0, actions_max_len - actions_len], [0, 0]])

    reshaped_padded_buffer = tf.reshape(padded_buffer, [1, buffer_max_len, lstm_dim])
    reshaped_padded_stack = tf.reshape(padded_stack, [1, stack_max_len, lstm_dim])
    reshaped_padded_actions = tf.reshape(padded_actions, [1, actions_max_len, lstm_dim])

    lstm_buffer = nn_run_lstm_input(reshaped_padded_buffer, lstm_dim, buffer_len, 'lstm_buffer')
    lstm_stack = nn_run_lstm_input(reshaped_padded_stack, lstm_dim, stack_len, 'lstm_stack')
    lstm_actions = nn_run_lstm_input(reshaped_padded_actions, lstm_dim, actions_len, 'lstm_actions')

    input_concat_vec = tf.concat([lstm_stack, lstm_buffer, lstm_actions], axis=-1)
    input_dim = 3 * lstm_dim
    input_weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1.0/sqrt(output_dim)),
                                name='weight_input')
    input_bias = tf.Variable(tf.zeros(output_dim), name='bias_input')
    input_out = tf.nn.relu(tf.matmul(input_concat_vec, input_weight) + input_bias)

with tf.name_scope('output_layer'):
    output_weight = tf.Variable(tf.truncated_normal([output_dim, n_class], stddev=1.0/sqrt(n_class)),
                                name='weight_output')
    output_bias = tf.Variable(tf.zeros(n_class), name='bias_output')
    out = tf.matmul(input_out, output_weight) + output_bias

    prediction = tf.multiply(tf.nn.softmax(out), output_mask)

with tf.name_scope('calculate_loss'):
    dropped_hidden = tf.nn.dropout(input_out, dropout_prob)
    dropped_output = tf.matmul(dropped_hidden, output_weight) + output_bias

    one_hot_labels = tf.one_hot(label, n_class, on_value=1.0, off_value=0.0)
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dropped_output, labels=one_hot_labels))
    loss = ce_loss

with tf.name_scope('initial_buffer'):
    mapped_subwords_b = tf.nn.embedding_lookup(subword_emb, subword_ph)
    mapped_word_can = tf.nn.embedding_lookup(word_emb, word_candidates_ph)
    reshaped_word_can = tf.reshape(mapped_word_can, [tf.shape(mapped_word_can)[0], -1])
    mapped_bpos = tf.nn.embedding_lookup(bpos_emb, bpos_ph)
    reshaped_bpos = tf.reshape(mapped_bpos, [tf.shape(mapped_bpos)[0], -1])

    buffer_concat_vec = tf.concat([mapped_subwords_b, reshaped_word_can, reshaped_bpos], axis=-1)

    input_dim = ((k_word_candidate + k_bpos_candidate) * candidate_emb_dim) + word_emb_dim
    buffer_weight = tf.Variable(tf.truncated_normal([input_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)),
                                name='weight_buffer')
    buffer_bias = tf.Variable(tf.zeros([lstm_dim]), name='bias_buffer')
    buffer_list = tf.nn.relu(tf.matmul(buffer_concat_vec, buffer_weight) + buffer_bias)
    reshaped_buffer = tf.expand_dims(buffer_list, 0)

    assign_buffer = tf.assign(buffer, reshaped_buffer, validate_shape=False)

with tf.name_scope('remove_buffer'):
    shorten_buffer = buffer[:, :-1, :]
    remove_from_buffer = tf.assign(buffer, shorten_buffer, validate_shape=False)

with tf.name_scope('stack_config'):
    stack_word_dim = word_emb_dim + subword_lstm_dim + param_emb_dim
    stack_word_weight = tf.Variable(tf.truncated_normal([stack_word_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)),
                                    name='weight_stack_word')
    stack_word_bias = tf.Variable(tf.zeros([lstm_dim]), name='bias_stack_word')

    stack_rel_dim = (2 * lstm_dim) + param_emb_dim
    stack_rel_weight = tf.Variable(tf.truncated_normal([stack_rel_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)),
                                   name='weight_stack_rel')
    stack_rel_bias = tf.Variable(tf.zeros([lstm_dim]), name='bias_stack_rel')

with tf.name_scope('action_config'):
    action_weight = tf.Variable(tf.truncated_normal([param_emb_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)),
                                name='weight_action')
    action_bias = tf.Variable(tf.zeros(lstm_dim), name='bias_action')

with tf.name_scope('stack_word_node'):
    mapped_word = tf.nn.embedding_lookup(word_lm_emb, word_ph)
    mapped_subwords_s = tf.nn.embedding_lookup(subword_emb, subword_list_ph)
    subword_lstm = nn_run_lstm_input(tf.expand_dims(mapped_subwords_s, 0), subword_lstm_dim, subwords_len, 'lstm_subword')
    mapped_pos = tf.nn.embedding_lookup(pos_emb, pos_ph)

    stack_concat_vec = tf.concat([mapped_word, subword_lstm, mapped_pos], axis=-1)
    stack_vec = tf.nn.relu(tf.matmul(stack_concat_vec, stack_word_weight) + stack_word_bias)
    reshaped_stack_vec = tf.expand_dims(stack_vec, 0)

with tf.name_scope('left_arc'):
    left_dep_node = stack[:, -2, :]
    right_head_node = stack[:, -1, :]
    la_mapped_action = tf.nn.embedding_lookup(action_emb, relation_action_ph)
    left_arc_concat = tf.concat([right_head_node, left_dep_node, la_mapped_action], axis=-1)
    composed_left_arc = tf.nn.tanh(tf.matmul(left_arc_concat, stack_rel_weight) + stack_rel_bias)
    reshaped_left_arc_vec = tf.expand_dims(composed_left_arc, 0)

with tf.name_scope('right_arc'):
    left_head_node = stack[:, -2, :]
    right_dep_node = stack[:, -1, :]
    ra_mapped_action = tf.nn.embedding_lookup(action_emb, relation_action_ph)
    right_arc_concat = tf.concat([left_head_node, right_dep_node, ra_mapped_action], axis=-1)
    composed_right_arc = tf.nn.tanh(tf.matmul(right_arc_concat, stack_rel_weight) + stack_rel_bias)
    reshaped_right_arc_vec = tf.expand_dims(composed_right_arc, 0)

with tf.name_scope('modify_stack'):
    # intialize stack
    initial_stack = tf.assign(stack, reshaped_stack_vec, validate_shape=False)

    # shift
    new_word_stack = tf.concat([stack, reshaped_stack_vec], axis=1)
    add_word_to_stack = tf.assign(stack, new_word_stack, validate_shape=False)

    # append
    new_replaced_word = tf.concat([stack[:, :-1, :], reshaped_stack_vec], axis=1)
    replace_word_tos = tf.assign(stack, new_replaced_word, validate_shape=False)

    # left-arc
    new_stack_left_arc = tf.concat([stack[:, :-2, :], reshaped_left_arc_vec], axis=1)
    left_arc_replace = tf.assign(stack, new_stack_left_arc, validate_shape=False)

    # right-arc
    new_stack_right_arc = tf.concat([stack[:, :-2, :], reshaped_right_arc_vec], axis=1)
    right_arc_replace = tf.assign(stack, new_stack_right_arc, validate_shape=False)

with tf.name_scope('modify_actions'):
    mapped_actions = tf.nn.embedding_lookup(action_emb, action_ph)
    action_vec = tf.nn.relu(tf.matmul(mapped_actions, action_weight) + action_bias)
    reshaped_action_vec = tf.expand_dims(action_vec, 0)

    initial_actions = tf.assign(actions, reshaped_action_vec, validate_shape=False)

    new_actions = tf.concat([actions, reshaped_action_vec], axis=1)
    add_to_actions = tf.assign(actions, new_actions, validate_shape=False)

optimize = tf.train.AdamOptimizer(name='parser_opt').minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=n_kept_model)


class ParserModel:
    def __init__(self, params, embeddings=None, model_path=None):
        self.params = params
        self.session = tf.Session()
        if model_path is not None:
            saver.restore(self.session, model_path)
        else:
            if embeddings is None:
                raise Exception('You must feed embeddings value if model_path is not defined.')

            feed_dict = {word_emb_ph: embeddings['word'], subword_emb_ph: embeddings['subword']}
            self.session.run(init)
            self.session.run([assign_word_embedding, assign_subword_embedding], feed_dict=feed_dict)

        self.subwords = list()
        self.tos_subwords = list()
        self.stack_len = 0
        self.buffer_len = 0
        self.actions_len = 0

    def train(self, action_label, feasible_actions):
        feed_dict = {
            label: [action_label],
            output_mask: feasible_actions,
            stack_len: self.stack_len,
            buffer_len: self.buffer_len,
            actions_len: self.actions_len
        }
        _, train_loss = self.session.run([optimize, loss], feed_dict=feed_dict)
        return train_loss

    def predict(self, feasible_actions):
        feed_dict = {
            output_mask: feasible_actions,
            stack_len: self.stack_len,
            buffer_len: self.buffer_len,
            actions_len: self.actions_len
        }
        pred = self.session.run(prediction, feed_dict=feed_dict)
        return argmax(pred)

    def save_model(self, model_path, global_step):
        saver.save(self.session, model_path, global_step=global_step)
        print('Model at epoch', global_step, 'is saved.')

    def initial_parser_model(self, subwords, word_candidates, bpos_candidates, real_subword):
        # reset subword list
        self.tos_subwords = list()
        self.subwords = real_subword
        self.buffer_len = len(real_subword) + 1
        self.stack_len = 1
        self.actions_len = 1

        # reset in network
        feed_dict = {
            subword_ph: [1] + list(reversed(subwords)),
            word_candidates_ph: [([1] * k_word_candidate)] + list(reversed(word_candidates)),
            bpos_ph: [([n_bpos - 1] * k_bpos_candidate)] + list(reversed(bpos_candidates)),
            buffer_len: self.buffer_len
        }
        self.session.run(assign_buffer, feed_dict=feed_dict)

        feed_dict = {
            word_ph: [1],
            subword_list_ph: [1],
            pos_ph: [n_pos - 1],
            stack_len: self.buffer_len,
            subwords_len: 1
        }
        self.session.run(initial_stack, feed_dict=feed_dict)
        self.session.run(initial_actions, feed_dict={action_ph: [n_action - 1], actions_len: self.actions_len})

    def take_action(self, action_index):
        (action, params) = self.params['reverse_action_map'][action_index]
        if action == 'SHIFT':
            self.take_action_shift(params)
        elif action == 'APPEND':
            self.take_action_append(params)
        elif action == 'LEFT-ARC':
            self.take_action_left_arc(action_index)
        else:
            self.take_action_right_arc(action_index)

        self.actions_len += 1
        feed_dict = {action_ph: [action_index], actions_len: self.actions_len}
        self.session.run(add_to_actions, feed_dict=feed_dict)

    def take_action_shift(self, pos_tag):
        word = self.subwords[0]
        self.tos_subwords = [word]
        self.subwords = self.subwords[1:]
        self.stack_len += 1
        self.buffer_len -= 1

        feed_dict = self.get_word_stack_feed_dict(word, self.tos_subwords, pos_tag)
        self.session.run([add_word_to_stack, remove_from_buffer], feed_dict=feed_dict)

    def take_action_append(self, pos_tag):
        subword = self.subwords[0]
        self.tos_subwords.append(subword)
        word = ''.join(self.tos_subwords)
        self.subwords = self.subwords[1:]
        self.buffer_len -= 1

        feed_dict = self.get_word_stack_feed_dict(word, self.tos_subwords, pos_tag)
        self.session.run([replace_word_tos, remove_from_buffer], feed_dict=feed_dict)

    def take_action_left_arc(self, action_index):
        self.stack_len -= 1
        feed_dict = {relation_action_ph: [action_index], stack_len: self.stack_len}
        self.session.run(left_arc_replace, feed_dict=feed_dict)

    def take_action_right_arc(self, action_index):
        self.stack_len -= 1
        feed_dict = {relation_action_ph: [action_index], stack_len: self.stack_len}
        self.session.run(right_arc_replace, feed_dict=feed_dict)

    def get_word_stack_feed_dict(self, word, subwords, pos):
        feed_dict = {
            word_ph: [self.params['word_map'].get(word, 0)],
            subword_list_ph: [self.params['subword_map'].get(subword, 0) for subword in subwords],
            pos_ph: [self.params['pos_map'][pos]],
            buffer_len: self.buffer_len,
            stack_len: self.buffer_len,
            subwords_len: len(self.subwords)
        }
        return feed_dict





