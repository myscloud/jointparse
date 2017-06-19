import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from math import sqrt
from numpy import argmax, zeros

# parameters
dropout_prob = 0.5
n_kept_model = 1

k_word_candidate = 3
k_bpos_candidate = 3

# dimension
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
n_class = 106
n_action = 107

# variables, placeholders, embeddings
buffer = tf.Variable(tf.zeros([1, lstm_dim]), name='buffer', trainable=False)
stack = tf.Variable(tf.zeros([2, lstm_dim]), name='stack', trainable=False)
actions = tf.Variable(tf.zeros([1, lstm_dim]), name='action', trainable=False)

buffer_lstm_vec = tf.Variable(tf.zeros([1, lstm_dim]), name='lstm_output_buffer', trainable=False)
stack_lstm_vec = tf.Variable(tf.zeros([1, lstm_dim]), name='lstm_output_stack', trainable=False)
action_lstm_vec = tf.Variable(tf.zeros([1, lstm_dim]), name='lstm_output_action', trainable=False)
stack_state = tf.Variable(tf.zeros([1, 2, lstm_dim]), name='state_stack', trainable=False)
actions_state = tf.Variable(tf.zeros([1, 2, lstm_dim]), name='state_actions', trainable=False)

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


def nn_run_lstm_input(input_vec, dim, scope_name, init_state=None):
    with tf.variable_scope(scope_name) as vs:
        lstm_fw_cell = rnn.BasicLSTMCell(dim, reuse=tf.get_variable_scope().reuse)

        if init_state is None:
            lstm_outputs, state = tf.nn.dynamic_rnn(lstm_fw_cell, input_vec, dtype=tf.float32, scope=vs)
        else:
            lstm_outputs, state = tf.nn.dynamic_rnn(lstm_fw_cell, input_vec, initial_state=init_state, scope=vs)
        final_outputs = tf.reshape(lstm_outputs[:, -1, :], [1, dim])
        final_state = tf.reshape(tf.squeeze(state), [1, 2, lstm_dim])
        return lstm_outputs, final_outputs, final_state

# run model
with tf.name_scope('input_layer'):
    input_concat_vec = tf.concat([stack_lstm_vec, buffer_lstm_vec, action_lstm_vec], axis=-1)
    input_dim = 3 * lstm_dim
    input_weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1.0/sqrt(output_dim)), name='weight_input')
    input_bias = tf.Variable(tf.zeros([output_dim]), name='bias_input')
    input_out = tf.nn.relu(tf.matmul(input_concat_vec, input_weight) + input_bias)

with tf.name_scope('output_layer'):
    output_weight = tf.Variable(tf.truncated_normal([output_dim, n_class], stddev=1.0 / sqrt(n_class)),
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

with tf.name_scope('optimize'):
    optimizer = tf.train.AdamOptimizer(name='parser_opt')
    compute_grad = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    computable_grad = [grad_info for grad_info in compute_grad if grad_info[0] is not None]
    gradients_list = [tf.Variable(tf.zeros(tf.shape(grad[1])), trainable=False) for grad in computable_grad]
    reset_gradient = [tf.assign(grad_sum, tf.zeros(tf.shape(grad[1])))
                      for grad_sum, grad in zip(gradients_list, computable_grad)]
    iter_compute_grad = [tf.assign(grad_sum, tf.add(grad_sum, grad[0]))
                         for grad_sum, grad in zip(gradients_list, computable_grad)]

    # gradient_tensor_ph = tf.placeholder(tf.float32, name='placeholder_gradient')
    # calc_gradient_sum = tf.reduce_sum(gradient_tensor_ph, 0)
    #
    # gradient_ph = [(tf.placeholder(tf.float32), grad_info[1]) for grad_info in compute_grad]
    epoch_grad = [(grad_sum, grad_info[1]) for grad_sum, grad_info in zip(gradients_list, computable_grad)]
    apply_grad = optimizer.apply_gradients(epoch_grad)

# stacks' node
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
    _, subword_lstm, _ = nn_run_lstm_input(tf.expand_dims(mapped_subwords_s, 0), subword_lstm_dim, 'lstm_subword')
    mapped_pos = tf.nn.embedding_lookup(pos_emb, pos_ph)

    stack_concat_vec = tf.concat([mapped_word, subword_lstm, mapped_pos], axis=-1)
    stack_vec = tf.nn.relu(tf.matmul(stack_concat_vec, stack_word_weight) + stack_word_bias)
    # reshaped_stack_vec = tf.expand_dims(stack_vec, 0)

with tf.name_scope('action_node'):
    mapped_action = tf.nn.embedding_lookup(action_emb, action_ph)
    reshaped_action = tf.reshape(mapped_action, [1, param_emb_dim])
    action_vec = tf.nn.relu(tf.matmul(reshaped_action, action_weight) + action_bias)
    # reshaped_action_vec = tf.expand_dims(action_vec, 0)

# initial stacks
with tf.name_scope('initial_buffer'):
    mapped_subwords_b = tf.nn.embedding_lookup(subword_emb, subword_ph)
    mapped_word_can = tf.nn.embedding_lookup(word_emb, word_candidates_ph)
    reshaped_word_can = tf.reshape(mapped_word_can, [tf.shape(mapped_word_can)[0], -1])
    mapped_bpos = tf.nn.embedding_lookup(bpos_emb, bpos_ph)
    reshaped_bpos = tf.reshape(mapped_bpos, [tf.shape(mapped_bpos)[0], -1])

    buffer_concat_vec = tf.concat([mapped_subwords_b, reshaped_word_can, reshaped_bpos], axis=-1)

    input_dim = word_emb_dim + ((k_word_candidate + k_bpos_candidate) * candidate_emb_dim)
    buffer_weight = tf.Variable(tf.truncated_normal([input_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)), name='weight_buffer')
    buffer_bias = tf.Variable(tf.zeros([lstm_dim]), name='bias_buffer')
    buffer_list = tf.nn.relu(tf.matmul(buffer_concat_vec, buffer_weight) + buffer_bias)
    reshaped_buffer_list = tf.expand_dims(buffer_list, 0)

    buffer_lstm_outputs, _, _ = nn_run_lstm_input(reshaped_buffer_list, lstm_dim, 'buffer_lstm')
    reshaped_buffer = tf.reshape(buffer_lstm_outputs, [tf.shape(buffer_lstm_outputs)[1], lstm_dim])
    initial_buffer = tf.assign(buffer, reshaped_buffer, validate_shape=False)

with tf.name_scope('initial_stack'):
    init_stack = tf.assign(stack, stack_vec, validate_shape=False)
    init_stack_state = tf.assign(stack_state, tf.zeros([1, 2, lstm_dim]), validate_shape=False)

with tf.name_scope('initial_action_stack'):
    init_action = tf.assign(actions, action_vec, validate_shape=False)
    init_action_state = tf.assign(actions_state, tf.zeros([1, 2, lstm_dim]), validate_shape=False)

# parser operation
with tf.name_scope('add_action'):
    add_action = tf.assign(actions, action_vec)

with tf.name_scope('buffer_remove'):
    remove_buffer = tf.assign(buffer, buffer[:-1, :], validate_shape=False)

with tf.name_scope('stack_shift'):
    shift_new_stack = tf.concat([stack, stack_vec], axis=0)
    shift_to_stack = tf.assign(stack, shift_new_stack, validate_shape=False)

with tf.name_scope('stack_append'):
    append_assign_new_state = tf.assign(stack_state, stack_state[:-1, :, :], validate_shape=False)
    append_new_stack = tf.concat([stack[:-1, :], stack_vec], axis=0)
    append_to_stack = tf.assign(stack, append_new_stack, validate_shape=False)

with tf.name_scope('stack_left_arc'):
    left_dep_node = tf.reshape(stack[-2, :], [1, lstm_dim])
    right_head_node = tf.reshape(stack[-1, :], [1, lstm_dim])
    la_mapped_action = tf.nn.embedding_lookup(action_emb, relation_action_ph)
    left_arc_concat = tf.concat([right_head_node, left_dep_node, la_mapped_action], axis=-1)
    left_arc_composed = tf.nn.tanh(tf.matmul(left_arc_concat, stack_rel_weight) + stack_rel_bias)

    la_assign_new_state = tf.assign(stack_state, stack_state[:-2, :, :], validate_shape=False)
    left_arc_new_stack = tf.concat([stack[:-2, :], left_arc_composed], axis=0)
    add_left_arc = tf.assign(stack, left_arc_new_stack, validate_shape=False)

with tf.name_scope('stack_right_arc'):
    left_head_node = tf.reshape(stack[-2, :], [1, lstm_dim])
    right_dep_node = tf.reshape(stack[-1, :], [1, lstm_dim])
    ra_mapped_action = tf.nn.embedding_lookup(action_emb, relation_action_ph)
    right_arc_concat = tf.concat([left_head_node, right_dep_node, ra_mapped_action], axis=-1)
    right_arc_composed = tf.nn.tanh(tf.matmul(right_arc_concat, stack_rel_weight) + stack_rel_bias)

    ra_assign_new_state = tf.assign(stack_state, stack_state[:-2, :, :], validate_shape=False)
    right_arc_new_stack = tf.concat([stack[:-2, :], right_arc_composed], axis=0)
    add_right_arc = tf.assign(stack, right_arc_new_stack, validate_shape=False)

with tf.name_scope('calculate_lstm_output'):
    calc_buffer_lstm = tf.assign(buffer_lstm_vec, tf.expand_dims(buffer[-1, :], 0))

    stack_last_state = rnn.LSTMStateTuple(
        tf.reshape(stack_state[-1, 0, :], [1, lstm_dim]), tf.reshape(stack_state[-1, 1, :], [1, lstm_dim]))
    reshaped_stack = tf.reshape(stack[-1, :], [1, 1, lstm_dim])
    _, stack_out, new_stack_state = nn_run_lstm_input(reshaped_stack, lstm_dim, 'stack_lstm', init_state=stack_last_state)
    calc_stack_lstm = tf.assign(stack_lstm_vec, stack_out)
    assign_stack_state = tf.assign(stack_state, tf.concat([stack_state, new_stack_state], axis=0), validate_shape=False)

    action_last_state = rnn.LSTMStateTuple(
        tf.reshape(actions_state[0, 0, :], [1, lstm_dim]), tf.reshape(actions_state[0, 1, :], [1, lstm_dim]))
    reshaped_actions = tf.reshape(actions, [1, 1, lstm_dim])
    _, action_out, new_action_state = nn_run_lstm_input(reshaped_actions, lstm_dim, 'action_lstm', init_state=action_last_state)
    calc_action_lstm = tf.assign(action_lstm_vec, action_out)
    assign_action_state = tf.assign(actions_state, new_action_state)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=n_kept_model)


class ParserModel:
    def __init__(self, params, embeddings=None, model_path=None):
        self.session = tf.Session()
        if model_path is not None:
            saver.restore(self.session, model_path)
        else:
            if embeddings is None:
                raise Exception('You have to feed embeddings to model if no model_path is identified.')

            feed_dict = {word_emb_ph: embeddings['word'], subword_emb_ph: embeddings['subword']}
            self.session.run(init)
            self.session.run([assign_word_embedding, assign_subword_embedding], feed_dict=feed_dict)

        self.params = params
        self.subwords = list()
        self.tos_subwords = list()
        self.gradients = [list() for _ in range(len(compute_grad))]
        self.grad_op = [grad[0] for grad in compute_grad if grad[0] is not None]
        self.grad_idx = [idx for idx, grad in enumerate(compute_grad) if grad[0] is not None]

        self.stack_states = list()
        self.action_state = None

    def calc_loss(self, action_label, feasible_actions):
        feed_dict = {
            label: [action_label],
            output_mask: feasible_actions
        }

        train_loss, _ = self.session.run([loss, iter_compute_grad], feed_dict=feed_dict)
        return train_loss

    def train(self):
        self.session.run(apply_grad)

    def predict(self, feasible_actions):
        pred = self.session.run(prediction, feed_dict={output_mask: feasible_actions})
        return argmax(pred)

    def save_model(self, model_path, global_step):
        saver.save(self.session, model_path, global_step=global_step)
        print('Model at epoch', global_step, 'is saved.')

    def initial_parser_model(self, subwords, word_candidates, bpos_candidates, real_subword):
        self.tos_subwords = list()
        self.subwords = real_subword
        self.stack_states = [rnn.LSTMStateTuple(zeros((1, lstm_dim)), zeros((1, lstm_dim)))]
        self.action_state = rnn.LSTMStateTuple(zeros((1, lstm_dim)), zeros((1, lstm_dim)))

        for grad_list in self.gradients:
            grad_list.clear()

        feed_dict = {
            # for buffer
            subword_ph: [1] + list(reversed(subwords)),
            word_candidates_ph: [([1] * k_word_candidate)] + list(reversed(word_candidates)),
            bpos_ph: [([n_bpos - 1] * k_bpos_candidate)] + list(reversed(bpos_candidates)),
            # for stack
            word_ph: [1],
            subword_list_ph: [1],
            pos_ph: [n_pos - 1],
            # for action
            action_ph: [n_action - 1]
        }
        init_action_list = [initial_buffer, init_stack, init_action,
                            init_stack_state, init_action_state, reset_gradient]
        self.session.run(init_action_list, feed_dict=feed_dict)
        self.calculate_lstm_vec()

    def take_action(self, action_index):
        (action, params) = self.params['reverse_action_map'][action_index]

        # generate new stack cell depends on action
        if action == 'SHIFT':
            self.take_action_shift(params)
        elif action == 'APPEND':
            self.take_action_append(params)
        elif action == 'LEFT-ARC':
            self.take_action_left_arc(action_index)
        else:
            self.take_action_right_arc(action_index)

        self.session.run(add_action, feed_dict={action_ph: action_index})
        self.calculate_lstm_vec()

    def calculate_lstm_vec(self):
        # generate action cell and lstm output in the graph
        action_list = [calc_buffer_lstm, calc_stack_lstm, calc_action_lstm, assign_stack_state, assign_action_state]
        self.session.run(action_list)

    def take_action_shift(self, pos_tag):
        word = self.subwords[0]
        self.tos_subwords = [word]
        self.subwords = self.subwords[1:]

        feed_dict = self.get_word_stack_feed_dict(word, self.tos_subwords, pos_tag)
        self.session.run([remove_buffer, shift_to_stack], feed_dict=feed_dict)

    def take_action_append(self, pos_tag):
        subword = self.subwords[0]
        self.tos_subwords.append(subword)
        word = ''.join(self.tos_subwords)
        self.subwords = self.subwords[1:]
        self.stack_states = self.stack_states[:-1]

        feed_dict = self.get_word_stack_feed_dict(word, self.tos_subwords, pos_tag)
        self.session.run([remove_buffer, append_to_stack, append_assign_new_state], feed_dict=feed_dict)

    def take_action_left_arc(self, action_index):
        self.stack_states = self.stack_states[:-2]
        self.session.run([add_left_arc, la_assign_new_state], feed_dict={relation_action_ph: [action_index]})

    def take_action_right_arc(self, action_index):
        self.stack_states = self.stack_states[:-2]
        self.session.run([add_right_arc, ra_assign_new_state], feed_dict={relation_action_ph: [action_index]})

    def get_word_stack_feed_dict(self, word, subwords, pos):
        feed_dict = {
            word_ph: [self.params['word_map'].get(word, 0)],
            subword_list_ph: [self.params['subword_map'].get(subword, 0) for subword in subwords],
            pos_ph: [self.params['pos_map'][pos]]
        }
        return feed_dict

