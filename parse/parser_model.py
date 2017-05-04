import tensorflow as tf
from math import sqrt

# parameters
feature_categories = ['word', 'subword', 'pos', 'label', 'bpos']
n_features = {'word': 21, 'subword': 3, 'pos': 15, 'label': 12, 'bpos': 6}
n_class = {'word': 100004, 'subword': 100004, 'pos': 16, 'label': 39, 'bpos': 61}
n_output_class = 106
embedding_dim = 64
n_hidden = 100

batch_size = 128
learning_rate = 0.5
dropout_prob = 0.5

with tf.device("/cpu:0"):
    x = dict()
    for feature in feature_categories:
        x[feature] = tf.placeholder(tf.int32, [batch_size, n_features[feature]])

    embedding = {
        'word': tf.placeholder(tf.float32, [n_class['word'], embedding_dim]),
        'subword': tf.placeholder(tf.float32, [n_class['subword'], embedding_dim]),
        'pos': tf.Variable(tf.random_uniform([n_class['pos'], embedding_dim], minval=-0.01, maxval=0.01)),
        'label': tf.Variable(tf.random_uniform([n_class['label'], embedding_dim], minval=-0.01, maxval=0.01)),
        'bpos': tf.Variable(tf.random_uniform([n_class['bpos'], embedding_dim], minval=-0.01, maxval=-0.01))
    }

    hidden_sum = tf.Variable(tf.zeros([n_hidden]))
    hidden_bias = tf.Variable(tf.zeros([n_hidden]))
    x_input = dict()
    flatten_x = dict()
    hidden_weights = dict()
    for keyword in x:
        x_input[keyword] = tf.nn.embedding_lookup(embedding[keyword], x[keyword])
        flatten_dim = n_features[keyword] * embedding_dim
        flatten_x[keyword] = tf.reshape(x_input[keyword], [batch_size, flatten_dim])
        hidden_weights[keyword] = tf.Variable(tf.truncated_normal([flatten_dim, n_hidden],
                                                                  stddev=1.0/sqrt(n_hidden)))
        hidden_sum = hidden_sum + tf.matmul(flatten_x[keyword], hidden_weights[keyword])

    hidden_output = tf.tanh(tf.add(hidden_sum, hidden_bias))

    output_weight = tf.Variable(tf.truncated_normal([n_hidden, n_output_class], stddev=1.0/sqrt(n_output_class)))
    output_bias = tf.Variable(tf.zeros([n_output_class]))
    outputs = tf.matmul(hidden_output, output_weight) + output_bias

    # for output
    min_value = tf.minimum(tf.reduce_min(outputs), 0.)
    positive_outputs = tf.add(outputs, (-1 * min_value))
    output_mask = tf.placeholder(tf.float32, [batch_size, n_output_class])
    final_outputs = tf.multiply(positive_outputs, output_mask)

    # optimize
    p_dropout = tf.placeholder(tf.float32)
    dropped_outputs = tf.nn.dropout(outputs, p_dropout)
    y = tf.placeholder(tf.int32, [batch_size, 1])
    y_vec = tf.one_hot(y, n_output_class, on_value=1.0, off_value=0.0, axis=-1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_vec, logits=dropped_outputs))
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


class ParserModel:
    def __init__(self, input_embedding, model_path=None):
        self.embedding = input_embedding
        self.session = tf.Session()
        if model_path is not None:
            saver.restore(self.session, model_path)
        else:
            self.session.run(init)

    def train(self, input_dict, labels, action_mask):
        feed_dict = self.get_feed_dict(input_dict, action_mask, labels=labels, keep_dropout=True)
        _, batch_loss = self.session.run([optimizer, loss], feed_dict=feed_dict)

        return batch_loss

    def evaluate(self, input_dict, labels, action_mask):
        feed_dict = self.get_feed_dict(input_dict, action_mask, labels=labels)
        batch_loss, results = self.session.run([loss, final_outputs], feed_dict=feed_dict)

        return batch_loss, results

    def predict(self, input_dict, action_mask):
        feed_dict = self.get_feed_dict(input_dict, action_mask)
        results = self.session.run(final_outputs, feed_dict=feed_dict)
        return results

    def get_feed_dict(self, input_dict, action_mask, labels=None, keep_dropout=False):
        feed_dict = dict()
        for feature in feature_categories:
            feed_dict[x[feature]] = input_dict[feature]
        feed_dict[output_mask] = action_mask
        feed_dict[embedding['word']] = self.embedding['word']
        feed_dict[embedding['subword']] = self.embedding['subword']
        feed_dict[p_dropout] = dropout_prob if keep_dropout else 0.0

        if labels is not None:
            feed_dict[y] = labels

        return feed_dict

    def save_model(self, save_path, step_no):
        saver.save(self.session, save_path, global_step=step_no)
