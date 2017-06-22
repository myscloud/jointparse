import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from math import sqrt
from numpy import argmax, zeros

# parameters
dropout_prob = 0.7
n_kept_model = 1

n_lstm_stack = 2
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
candidate_emb_dim = 30

cell_dim = lstm_dim + word_emb_dim
buffer_out_dim = word_emb_dim + (k_word_candidate * word_emb_dim) + (k_bpos_candidate * candidate_emb_dim)
stack_out_dim = (word_emb_dim * 2) + param_emb_dim

n_pos = 16
n_bpos = 61
# n_class = 106
n_class = {'action': 4, 'pos': 16, 'dep_label': 39}
n_action = 107
action_name_list = ['LEFT-ARC', 'RIGHT-ARC', 'SHIFT', 'APPEND']

# variables, placeholders, embeddings
buffer = tf.Variable(tf.zeros([1, lstm_dim]), name='buffer', trainable=False)
stack = tf.Variable(tf.zeros([2, cell_dim]), name='stack', trainable=False)
actions = tf.Variable(tf.zeros([1, lstm_dim]), name='action', trainable=False)

buffer_out = tf.Variable(tf.zeros([2, buffer_out_dim]), name='buffer_out', trainable=False)
stack_out = tf.Variable(tf.zeros([3, stack_out_dim]), name='stack_out', trainable=False)

buffer_lstm_vec = tf.Variable(tf.zeros([1, lstm_dim]), name='lstm_output_buffer', trainable=False)
stack_lstm_vec = tf.Variable(tf.zeros([1, lstm_dim]), name='lstm_output_stack', trainable=False)
action_lstm_vec = tf.Variable(tf.zeros([1, lstm_dim]), name='lstm_output_action', trainable=False)
stack_state = tf.Variable(tf.zeros([1, n_lstm_stack, 2, lstm_dim]), name='state_stack', trainable=False)
actions_state = tf.Variable(tf.zeros([1, n_lstm_stack, 2, lstm_dim]), name='state_actions', trainable=False)

subword_ph = tf.placeholder(tf.int32, [None], name='placeholder_subword')
word_candidates_ph = tf.placeholder(tf.int32, [None, None], name='placeholder_word_candidates')
bpos_ph = tf.placeholder(tf.int32, [None, None], name='placeholder_bpos')
word_ph = tf.placeholder(tf.int32, None, name='placeholder_word')
subword_list_ph = tf.placeholder(tf.int32, [None], name='placeholder_subword_list')
pos_ph = tf.placeholder(tf.int32, None, name='placeholder_pos')
relation_action_ph = tf.placeholder(tf.int32, [None], name='placeholder_rel_action')
action_ph = tf.placeholder(tf.int32, None, name='placeholder_action')
output_mask = tf.placeholder(tf.float32, [None], name='placeholder_output_mask')
label_ph = {
    'action': tf.placeholder(tf.int32, None, name='placeholder_action_label'),
    'pos': tf.placeholder(tf.int32, None, name='placeholder_pos_label'),
    'dep_label': tf.placeholder(tf.int32, None, name='placeholder_dep_label'),
}

word_lm_emb = tf.Variable(tf.zeros([word_vocab_size, word_emb_dim]), trainable=False, name='embedding_word_lm')
subword_emb = tf.Variable(tf.zeros([subword_vocab_size, word_emb_dim]), trainable=False, name='embedding_subword')
word_emb_ph = tf.placeholder(tf.float32, [word_vocab_size, word_emb_dim])
subword_emb_ph = tf.placeholder(tf.float32, [word_vocab_size, word_emb_dim])
assign_word_embedding = tf.assign(word_lm_emb, word_emb_ph, validate_shape=False)
assign_subword_embedding = tf.assign(subword_emb, subword_emb_ph, validate_shape=False)

word_emb = tf.Variable(tf.random_uniform([word_vocab_size, word_emb_dim], minval=-0.1, maxval=0.1),
                       name='embedding_word')
bpos_emb = tf.Variable(tf.random_uniform([n_bpos, candidate_emb_dim], minval=-0.1, maxval=0.1), name='embedding_bpos')

pos_emb = tf.Variable(tf.random_uniform([n_pos, param_emb_dim], minval=-0.1, maxval=0.1), name='embedding_pos')
action_emb = tf.Variable(tf.random_uniform([n_action, param_emb_dim], minval=-0.1, maxval=0.1), name='embedding_action')


def nn_run_lstm_input(input_vec, dim, scope_name, init_state=None):
    with tf.variable_scope(scope_name) as vs:
        # lstm_fw_cell = rnn.BasicLSTMCell(dim, reuse=tf.get_variable_scope().reuse)
        lstm_fw_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(dim, reuse=tf.get_variable_scope().reuse)
                                         for _ in range(n_lstm_stack)])

        if init_state is None:
            lstm_outputs, state = tf.nn.dynamic_rnn(lstm_fw_cell, input_vec, dtype=tf.float32, scope=vs)
        else:
            lstm_outputs, state = tf.nn.dynamic_rnn(lstm_fw_cell, input_vec, initial_state=init_state, scope=vs)

        # print(state)
        final_outputs = tf.reshape(lstm_outputs[:, -1, :], [1, dim])
        state_list = [tf.reshape(tf.concat([state[i].c, state[i].h], axis=-1), [1, 2, lstm_dim]) for i in range(n_lstm_stack)]
        final_state = tf.reshape(tf.concat(state_list, axis=0), [1, n_lstm_stack, 2, lstm_dim])
        return lstm_outputs, final_outputs, final_state

# run model
with tf.name_scope('input_layer'):
    # whole parser configuration
    input_concat_vec = tf.concat([stack_lstm_vec, buffer_lstm_vec, action_lstm_vec], axis=-1)
    input_dim = 3 * lstm_dim

    input_weight = tf.Variable(tf.truncated_normal([input_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)), name='weight_input')
    input_bias = tf.Variable(tf.zeros([lstm_dim]), name='bias_input')
    input_parser = tf.nn.relu(tf.matmul(input_concat_vec, input_weight) + input_bias)

    # top stack/buffer configuration
    top_stack_rel = tf.reshape(stack[-2:, :], [1, 2*cell_dim])
    top_stack_word = tf.reshape(stack_out[-2:, :], [1, 2*stack_out_dim])
    top_buffer = tf.reshape(buffer_out[0:2, :], [1, 2*buffer_out_dim])
    top_config = tf.concat([top_stack_rel, top_stack_word, top_buffer], axis=-1)
    config_input_dim = (2*cell_dim) + (2*stack_out_dim) + (2*buffer_out_dim)
    config_weight = tf.Variable(tf.truncated_normal([config_input_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)), name='weight_config')
    config_bias = tf.Variable(tf.zeros([lstm_dim]), name='bias_config')
    input_config = tf.nn.relu(tf.matmul(top_config, config_weight) + config_bias)
    input_out = tf.concat([input_parser, input_config], axis=-1)

with tf.name_scope('hidden_layer'):
    hidden_weight = tf.Variable(tf.truncated_normal([2*lstm_dim, output_dim], stddev=1.0/sqrt(output_dim)),
                                name='weight_hidden')
    hidden_bias = tf.Variable(tf.truncated_normal([output_dim]), name='bias_hidden')
    hidden_out = tf.nn.relu(tf.matmul(input_out, hidden_weight) + hidden_bias)

with tf.name_scope('output_layer'):
    output_weights = dict()
    output_bias = dict()
    predictions = dict()
    dropped = dict()

    dropped_hidden = tf.nn.dropout(hidden_out, dropout_prob)
    for output_name in n_class:
        output_weights[output_name] = tf.Variable(
            tf.truncated_normal([output_dim, n_class[output_name]], stddev=1.0/sqrt(n_class[output_name])),
            name='weight_output_'+output_name)
        output_bias[output_name] = tf.Variable(tf.zeros(n_class[output_name]), name='bias_output_' + output_name)
        predictions[output_name] = tf.matmul(hidden_out, output_weights[output_name]) + output_bias[output_name]
        dropped[output_name] = tf.matmul(dropped_hidden, output_weights[output_name]) + output_bias[output_name]

    predictions['action'] = tf.multiply(tf.nn.softmax(predictions['action']), output_mask)
    dropped['action'] = tf.multiply(dropped['action'], output_mask)

with tf.name_scope('calculate_loss'):
    loss = 0
    for output_name in n_class:
        one_hot_labels = tf.one_hot(label_ph[output_name], n_class[output_name], on_value=1.0, off_value=0.0)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dropped[output_name], labels=one_hot_labels))
        if output_name != 'action':
            ce_loss *= 0.25
        loss += ce_loss

with tf.name_scope('optimize'):
    optimizer = tf.train.AdamOptimizer(name='parser_opt', beta1=0.95)
    compute_grad = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    computable_grad = [grad_info for grad_info in compute_grad if grad_info[0] is not None]
    gradients_list = [tf.Variable(tf.zeros(tf.shape(grad[1])), trainable=False) for grad in computable_grad]
    reset_gradient = [tf.assign(grad_sum, tf.zeros(tf.shape(grad[1])))
                      for grad_sum, grad in zip(gradients_list, computable_grad)]

    iter_compute_grad = [tf.assign(grad_sum, tf.add(grad_sum, grad[0]))
                         for grad_sum, grad in zip(gradients_list, computable_grad)]

    epoch_grad = [(grad_sum, grad_info[1]) for grad_sum, grad_info in zip(gradients_list, computable_grad)]
    apply_grad = optimizer.apply_gradients(epoch_grad)

# stacks' node
with tf.name_scope('stack_config'):
    stack_word_dim = word_emb_dim + word_emb_dim + param_emb_dim
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
    mapped_word_lm = tf.nn.embedding_lookup(word_lm_emb, word_ph)
    mapped_subwords_s = tf.nn.embedding_lookup(subword_emb, subword_list_ph)
    _, subword_lstm, _ = nn_run_lstm_input(tf.expand_dims(mapped_subwords_s, 0), subword_lstm_dim, 'lstm_subword')
    mapped_action_s = tf.nn.embedding_lookup(action_emb, relation_action_ph)

    stack_concat_vec = tf.concat([mapped_word_lm, subword_lstm, mapped_action_s], axis=-1)
    assign_stack_out = tf.assign(stack_out, tf.concat([stack_out, stack_concat_vec], axis=0), validate_shape=False)

    stack_composed_vec = tf.nn.relu(tf.matmul(stack_concat_vec, stack_word_weight) + stack_word_bias)
    stack_vec = tf.concat([stack_composed_vec, mapped_word_lm], axis=-1)

with tf.name_scope('action_node'):
    mapped_action = tf.nn.embedding_lookup(action_emb, action_ph)
    reshaped_action = tf.reshape(mapped_action, [1, param_emb_dim])
    action_vec = tf.nn.relu(tf.matmul(reshaped_action, action_weight) + action_bias)

# initial stacks
with tf.name_scope('initial_buffer'):
    mapped_subwords_b = tf.nn.embedding_lookup(subword_emb, subword_ph)
    mapped_word_can = tf.nn.embedding_lookup(word_lm_emb, word_candidates_ph)
    reshaped_word_can = tf.reshape(mapped_word_can, [tf.shape(mapped_word_can)[0], -1])
    mapped_bpos = tf.nn.embedding_lookup(bpos_emb, bpos_ph)
    reshaped_bpos = tf.reshape(mapped_bpos, [tf.shape(mapped_bpos)[0], -1])

    buffer_concat_vec = tf.concat([mapped_subwords_b, reshaped_word_can, reshaped_bpos], axis=-1)
    init_buffer_out = tf.assign(buffer_out, buffer_concat_vec, validate_shape=False)
    buffer_weight = tf.Variable(tf.truncated_normal([buffer_out_dim, lstm_dim], stddev=1.0/sqrt(lstm_dim)), name='weight_buffer')
    buffer_bias = tf.Variable(tf.zeros([lstm_dim]), name='bias_buffer')
    buffer_list = tf.nn.relu(tf.matmul(buffer_concat_vec, buffer_weight) + buffer_bias)
    final_buffer_list = tf.concat([buffer_list, mapped_subwords_b], axis=-1)
    reshaped_buffer_list = tf.expand_dims(final_buffer_list, 0)

    buffer_lstm_outputs, _, _ = nn_run_lstm_input(reshaped_buffer_list, lstm_dim, 'buffer_lstm')
    reshaped_buffer = tf.reshape(buffer_lstm_outputs, [tf.shape(buffer_lstm_outputs)[1], lstm_dim])
    initial_buffer = tf.assign(buffer, reshaped_buffer, validate_shape=False)

with tf.name_scope('initial_stack'):
    init_stack = tf.assign(stack, stack_vec, validate_shape=False)
    init_stack_out = tf.assign(stack_out, stack_concat_vec, validate_shape=False)
    init_stack_state = tf.assign(stack_state, tf.zeros([1, n_lstm_stack, 2, lstm_dim]), validate_shape=False)

with tf.name_scope('initial_action_stack'):
    init_action = tf.assign(actions, action_vec, validate_shape=False)
    init_action_state = tf.assign(actions_state, tf.zeros([1, n_lstm_stack, 2, lstm_dim]), validate_shape=False)

# parser operation
with tf.name_scope('add_action'):
    add_action = tf.assign(actions, action_vec)

with tf.name_scope('buffer_remove'):
    remove_buffer = tf.assign(buffer, buffer[1:, :], validate_shape=False)
    remove_buffer_out = tf.assign(buffer_out, buffer_out[1:, :], validate_shape=False)

with tf.name_scope('stack_shift'):
    shift_new_stack = tf.concat([stack, stack_vec], axis=0)
    shift_to_stack = tf.assign(stack, shift_new_stack, validate_shape=False)

with tf.name_scope('stack_append'):
    append_assign_new_state = tf.assign(stack_state, stack_state[:-1, :, :, :], validate_shape=False)
    append_new_stack_out = tf.concat([stack_out[:-1, :], stack_concat_vec], axis=0)
    append_assign_stack_out = tf.assign(stack_out, append_new_stack_out, validate_shape=False)
    append_new_stack = tf.concat([stack[:-1, :], stack_vec], axis=0)
    append_to_stack = tf.assign(stack, append_new_stack, validate_shape=False)

with tf.name_scope('stack_left_arc'):
    left_dep_node = tf.reshape(stack[-2, 0:lstm_dim], [1, lstm_dim])
    right_head_node = tf.reshape(stack[-1, 0:lstm_dim], [1, lstm_dim])
    la_mapped_action = tf.nn.embedding_lookup(action_emb, relation_action_ph)
    left_arc_concat = tf.concat([right_head_node, left_dep_node, la_mapped_action], axis=-1)
    left_arc_composed = tf.nn.tanh(tf.matmul(left_arc_concat, stack_rel_weight) + stack_rel_bias)
    right_head_node_word = tf.reshape(stack[-1, lstm_dim:], [1, word_emb_dim])
    left_arc_vec = tf.concat([left_arc_composed, right_head_node_word], axis=-1)

    la_new_stack_out = tf.concat([stack_out[:-2, :], tf.expand_dims(stack_out[-1, :], 0)], axis=0)
    la_assign_stack_out = tf.assign(stack_out, la_new_stack_out, validate_shape=False)
    la_assign_new_state = tf.assign(stack_state, stack_state[:-2, :, :, :], validate_shape=False)
    left_arc_new_stack = tf.concat([stack[:-2, :], left_arc_vec], axis=0)
    add_left_arc = tf.assign(stack, left_arc_new_stack, validate_shape=False)

with tf.name_scope('stack_right_arc'):
    left_head_node = tf.reshape(stack[-2, 0:lstm_dim], [1, lstm_dim])
    right_dep_node = tf.reshape(stack[-1, 0:lstm_dim], [1, lstm_dim])
    ra_mapped_action = tf.nn.embedding_lookup(action_emb, relation_action_ph)
    right_arc_concat = tf.concat([left_head_node, right_dep_node, ra_mapped_action], axis=-1)
    right_arc_composed = tf.nn.tanh(tf.matmul(right_arc_concat, stack_rel_weight) + stack_rel_bias)
    left_head_node_word = tf.reshape(stack[-2, lstm_dim:], [1, word_emb_dim])
    right_arc_vec = tf.concat([right_arc_composed, left_head_node_word], axis=-1)

    ra_assign_stack_out = tf.assign(stack_out, stack_out[:-1, :], validate_shape=False)
    ra_assign_new_state = tf.assign(stack_state, stack_state[:-2, :, :, :], validate_shape=False)
    right_arc_new_stack = tf.concat([stack[:-2, :], right_arc_vec], axis=0)
    add_right_arc = tf.assign(stack, right_arc_new_stack, validate_shape=False)

with tf.name_scope('calculate_lstm_output'):
    calc_buffer_lstm = tf.assign(buffer_lstm_vec, tf.expand_dims(buffer[0, :], 0))

    stack_last_state = tuple([rnn.LSTMStateTuple(
        tf.reshape(stack_state[-1, i, 0, :], [1, lstm_dim]), tf.reshape(stack_state[-1, i, 1, :], [1, lstm_dim]))
                        for i in range(n_lstm_stack)])
    reshaped_stack = tf.reshape(stack[-1, :], [1, 1, cell_dim])
    _, stack_lstm_out, new_stack_state = nn_run_lstm_input(reshaped_stack, lstm_dim, 'stack_lstm', init_state=stack_last_state)
    calc_stack_lstm = tf.assign(stack_lstm_vec, stack_lstm_out)
    assign_stack_state = tf.assign(stack_state, tf.concat([stack_state, new_stack_state], axis=0), validate_shape=False)

    action_last_state = tuple([rnn.LSTMStateTuple(
        tf.reshape(actions_state[0, i, 0, :], [1, lstm_dim]), tf.reshape(actions_state[0, i, 1, :], [1, lstm_dim]))
                         for i in range(n_lstm_stack)])
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
        self.action_count = 0

    def calc_loss(self, action_label, feasible_actions):
        (action, params) = self.params['reverse_action_map'][action_label]
        action_label = action_name_list.index(action)
        pos_label = self.params['pos_map'][params] if action_label >= 2 else (n_class['pos'] - 1)
        dep_label = self.params['dep_label_map'][params] if action_label < 2 else (n_class['dep_label'] - 1)

        feed_dict = {
            label_ph['action']: [action_label],
            label_ph['pos']: [pos_label],
            label_ph['dep_label']: [dep_label],
            output_mask: feasible_actions
        }

        train_loss, _ = self.session.run([loss, iter_compute_grad], feed_dict=feed_dict)
        return train_loss

    def train(self):
        self.session.run(apply_grad)

    def predict(self, feasible_actions):
        predicted_outputs = self.session.run(predictions, feed_dict={output_mask: feasible_actions})
        max_action = argmax(predicted_outputs['action'])
        action = action_name_list[max_action]
        if max_action >= 2:
            params = self.params['reverse_pos_map'][argmax(predicted_outputs['pos'][0][:-1])]
        else:
            params = self.params['reverse_dep_label_map'][argmax(predicted_outputs['dep_label'][0][:-1])]
        action_index = self.params['action_map'][(action, params)]
        return action_index

    def save_model(self, model_path, global_step):
        saver.save(self.session, model_path, global_step=global_step)
        print('Model at epoch', global_step, 'is saved.')

    def initial_parser_model(self, subwords, word_candidates, bpos_candidates, real_subword):
        self.tos_subwords = list()
        self.subwords = real_subword
        self.action_count = 0

        for grad_list in self.gradients:
            grad_list.clear()

        feed_dict = {
            # for buffer
            subword_ph: subwords + [3, 3],
            word_candidates_ph: [([3] * k_word_candidate)] + [([3] * k_word_candidate)] + list(reversed(word_candidates)),
            bpos_ph: [([n_bpos - 1] * k_bpos_candidate)] + [([n_bpos - 1] * k_bpos_candidate)] + list(reversed(bpos_candidates)),
            # for stack
            word_ph: [3],
            subword_list_ph: [3],
            # pos_ph: [n_pos - 1],
            relation_action_ph: [82],  # SHIFT X
            # for action
            action_ph: [n_action - 1]
        }
        init_action_list = [initial_buffer, init_buffer_out, init_stack, init_action,
                            init_stack_state, init_stack_out, init_action_state, reset_gradient]
        self.session.run(init_action_list, feed_dict=feed_dict)

        root_feed_dict = {
            word_ph: [1],
            subword_list_ph: [1],
            relation_action_ph: [82]
        }
        self.session.run([shift_to_stack, assign_stack_out], feed_dict=root_feed_dict)
        self.calculate_lstm_vec()

    def take_action(self, action_index):
        (action, params) = self.params['reverse_action_map'][action_index]

        # generate new stack cell depends on action
        if action == 'SHIFT':
            self.take_action_shift(action_index)
        elif action == 'APPEND':
            self.take_action_append(action_index)
        elif action == 'LEFT-ARC':
            self.take_action_left_arc(action_index)
        else:
            self.take_action_right_arc(action_index)

        self.session.run(add_action, feed_dict={action_ph: action_index})
        self.calculate_lstm_vec()
        self.action_count += 1

    def calculate_lstm_vec(self):
        # generate action cell and lstm output in the graph
        action_list = [calc_buffer_lstm, calc_stack_lstm, calc_action_lstm, assign_stack_state, assign_action_state]
        self.session.run(action_list)

    def take_action_shift(self, action_index):
        word = self.subwords[0]
        self.tos_subwords = [word]
        self.subwords = self.subwords[1:]

        feed_dict = self.get_word_stack_feed_dict(word, self.tos_subwords, action_index)
        self.session.run([remove_buffer, remove_buffer_out, shift_to_stack, assign_stack_out], feed_dict=feed_dict)

    def take_action_append(self, action_index):
        subword = self.subwords[0]
        self.tos_subwords.append(subword)
        word = ''.join(self.tos_subwords)
        self.subwords = self.subwords[1:]

        feed_dict = self.get_word_stack_feed_dict(word, self.tos_subwords, action_index)
        action_list = [remove_buffer, remove_buffer_out, append_to_stack, append_assign_new_state, append_assign_stack_out]
        self.session.run(action_list, feed_dict=feed_dict)

    def take_action_left_arc(self, action_index):
        action_list = [add_left_arc, la_assign_new_state, la_assign_stack_out]
        self.session.run(action_list, feed_dict={relation_action_ph: [action_index]})

    def take_action_right_arc(self, action_index):
        action_list = [add_right_arc, ra_assign_new_state, ra_assign_stack_out]
        self.session.run(action_list, feed_dict={relation_action_ph: [action_index]})

    def get_word_stack_feed_dict(self, word, subwords, action_index):
        feed_dict = {
            word_ph: [self.params['word_map'].get(word, 0)],
            subword_list_ph: [self.params['subword_map'].get(subword, 0) for subword in subwords],
            relation_action_ph: [action_index]
        }
        return feed_dict

