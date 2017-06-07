import tensorflow as tf
from math import sqrt

# parameters
feature_categories = ['word', 'subword', 'pos', 'label', 'bpos']
n_features = {'word': 24, 'subword': 3, 'pos': 15, 'label': 12, 'bpos': 9}
n_class = {'word': 100004, 'subword': 100004, 'pos': 16, 'label': 39, 'bpos': 61}
n_output_class = 106
embedding_dim = 64
n_hidden = 150

batch_size = 64
learning_rate = 0.01
dropout_prob = 0.5
regularize_param = 10e-8

bn_epsilon = 10e-2
bn_decay = 0.9

n_kept_model = 2


def nn_batch_normalization(z, training_phase):
    bn_gamma = tf.Variable(tf.ones(z.get_shape()[-1]))
    bn_beta = tf.Variable(tf.zeros(z.get_shape()[-1]))
    pop_mean = tf.Variable(tf.zeros(z.get_shape()[-1]), trainable=False)
    pop_var = tf.Variable(tf.ones(z.get_shape()[-1]), trainable=False)

    batch_mean, batch_var = tf.nn.moments(z, [0])
    train_mean = tf.assign(pop_mean, tf.multiply(bn_decay, pop_mean) + tf.multiply(1-bn_decay, batch_mean))
    train_var = tf.assign(pop_var, tf.multiply(bn_decay, pop_var) + tf.multiply(1-bn_decay, batch_var))

    with tf.control_dependencies([train_mean, train_var]):
        def training_bn(): return tf.nn.batch_normalization(z, batch_mean, batch_var, bn_beta, bn_gamma, bn_epsilon)

    def test_bn(): return tf.nn.batch_normalization(z, pop_mean, pop_var, bn_beta, bn_gamma, bn_epsilon)

    return tf.cond(training_phase, training_bn, test_bn)


def nn_hidden_layer(x, embedding, phase):
    hidden_sum = tf.Variable(tf.zeros([n_hidden]))
    hidden_bias = tf.Variable(tf.zeros([n_hidden]))
    x_input = dict()
    flatten_x = dict()
    hidden_weights = dict()
    for keyword in x:
        x_input[keyword] = tf.nn.embedding_lookup(embedding[keyword], x[keyword])
        flatten_dim = n_features[keyword] * embedding_dim
        tensor_batch_size = tf.shape(x_input[keyword])[0]
        flatten_x[keyword] = tf.reshape(x_input[keyword], [tensor_batch_size, flatten_dim])
        hidden_weights[keyword] = tf.Variable(tf.truncated_normal([flatten_dim, n_hidden],
                                                                  stddev=1.0 / sqrt(n_hidden)))
        hidden_sum = hidden_sum + tf.matmul(flatten_x[keyword], hidden_weights[keyword])

    hidden_output = tf.pow(tf.add(hidden_sum, hidden_bias), 3)

    return hidden_output, hidden_weights, hidden_bias


def nn_output_layer(hidden_output, mask):
    output_weight = tf.Variable(tf.truncated_normal([n_hidden, n_output_class], stddev=1.0 / sqrt(n_output_class)))
    outputs = tf.matmul(hidden_output, output_weight)

    # for predicted output
    min_value = tf.minimum(tf.reduce_min(outputs), 0.)
    positive_outputs = tf.add(outputs, (-1 * min_value))
    final_outputs = tf.multiply(positive_outputs, mask)
    return final_outputs, output_weight


def nn_calculate_loss(y, outputs, parameters):
    # apply dropout
    dropped_outputs = tf.nn.dropout(outputs, dropout_prob)

    # cross entropy loss
    y_vec = tf.one_hot(y, n_output_class, on_value=1.0, off_value=0.0, axis=-1)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_vec, logits=dropped_outputs))

    # l2 regularization
    l2_sum = 0.0
    for param in parameters:
        l2_sum += tf.nn.l2_loss(param)
    l2_value = regularize_param * l2_sum

    # optimize
    loss = cross_entropy_loss + l2_value
    return loss

with tf.device('/cpu:0'):
    training_phase = tf.placeholder(tf.bool, name='training_phase')

    input_x = dict()
    for feature in feature_categories:
        input_x[feature] = tf.placeholder(tf.int32, [None, n_features[feature]], name='input_'+feature)

    input_embedding = {
        'word': tf.placeholder(tf.float32, [n_class['word'], embedding_dim], name='emb_word'),
        'subword': tf.placeholder(tf.float32, [n_class['subword'], embedding_dim], name='emb_subword'),
        'pos': tf.Variable(tf.random_uniform([n_class['pos'], embedding_dim], minval=-0.1, maxval=0.1)),
        'label': tf.Variable(tf.random_uniform([n_class['label'], embedding_dim], minval=-0.1, maxval=0.1)),
        'bpos': tf.Variable(tf.random_uniform([n_class['bpos'], embedding_dim], minval=-0.1, maxval=-0.1))
    }

    y_label = tf.placeholder(tf.int32, [None, 1], name='y')
    output_mask = tf.placeholder(tf.float32, [None, n_output_class], name='output_mask')

    h, h_weights, h_bias = nn_hidden_layer(input_x, input_embedding, training_phase)
    p, o_weight = nn_output_layer(h, output_mask)
    params = list(h_weights.values()) + [h_bias, o_weight]
    network_loss = nn_calculate_loss(y_label, p, params)

    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(network_loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=n_kept_model)


class ParserModel:
    def __init__(self, input_embedding, model_path=None):
        self.embedding = input_embedding
        self.session = tf.Session()
        self.options = dict()
        if model_path is not None:
            saver.restore(self.session, model_path)
        else:
            self.session.run(init)

    def train(self, input_dict, labels, action_mask):
        feed_dict = self.get_feed_dict(input_dict, action_mask, labels=labels, training=True)
        _, batch_loss = self.session.run([optimizer, network_loss], feed_dict=feed_dict)

        return batch_loss

    def evaluate(self, input_dict, labels, action_mask):
        feed_dict = self.get_feed_dict(input_dict, action_mask, labels=labels)
        batch_loss, results = self.session.run([network_loss, p], feed_dict=feed_dict)

        return batch_loss, results

    def predict(self, input_dict, action_mask):
        feed_dict = self.get_feed_dict(input_dict, action_mask)
        results = self.session.run(p, feed_dict=feed_dict)
        return results

    def get_feed_dict(self, input_dict, action_mask, labels=None, training=False):
        feed_dict = dict()
        for feature in feature_categories:
            feed_dict[input_x[feature]] = input_dict[feature]
        feed_dict[output_mask] = action_mask
        feed_dict[input_embedding['word']] = self.embedding['word']
        feed_dict[input_embedding['subword']] = self.embedding['subword']
        feed_dict[training_phase] = training

        if labels is not None:
            feed_dict[y_label] = labels

        return feed_dict

    def save_model(self, save_path, step_no):
        saver.save(self.session, save_path, global_step=step_no)
        print('Model at epoch', step_no, 'is saved.')
