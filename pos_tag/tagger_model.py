import tensorflow as tf
from tensorflow.contrib import rnn
from math import sqrt

# parameters
embedding_dim = 64
batch_size = 128
learning_rate = 0.01
vocabulary_size = 100004

# network parameters
n_steps = 30
n_input = 64
input_dim = 100
n_hidden_1 = 150
n_hidden_2 = 100
n_classes = 15
dropout_prob = 0.5
l2_beta = 10e-4

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.int32, [batch_size, n_steps])
    y = tf.placeholder(tf.int32, [batch_size, n_steps])
    sentence_len = tf.placeholder(tf.int32, [batch_size])
    embedding = tf.placeholder("float", [vocabulary_size, n_input])

    with tf.device('/cpu:0'):
        hidden_weights = tf.Variable(tf.truncated_normal([2 * n_hidden_1, n_hidden_2], stddev=1.0/sqrt(n_hidden_2)))
        hidden_biases = tf.Variable(tf.zeros([n_hidden_2]))

        out_weights = tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=1.0/sqrt(n_classes)))
        out_biases = tf.Variable(tf.zeros([n_classes]))

        x_input = tf.nn.embedding_lookup(embedding, x)
        x_input = tf.transpose(x_input, [1, 0, 2])
        x_input = tf.reshape(x_input, [-1, n_input])
        x_input = tf.split(x_input, n_steps, 0)

        y_label = tf.one_hot(y, depth=n_classes, on_value=1.0, off_value=0.0, axis=-1)
        y_label = tf.transpose(y_label, [1, 0, 2])
        y_label = tf.reshape(y_label, [-1, n_classes])
        y_label = tf.split(y_label, n_steps, 0)

        lstm_fw_cells = rnn.MultiRNNCell(cells=[rnn.BasicLSTMCell(n_hidden_1) for _ in range(2)])
        lstm_bw_cells = rnn.MultiRNNCell(cells=[rnn.BasicLSTMCell(n_hidden_1) for _ in range(2)])

        rnn_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cells, lstm_bw_cells, x_input,
                                                     dtype=tf.float32, sequence_length=sentence_len)

        hidden_outputs = [tf.nn.tanh(tf.matmul(rnn_outputs[idx], hidden_weights) + hidden_biases)
                          for idx in range(n_steps)]

        dropped_outputs = tf.nn.dropout(hidden_outputs, dropout_prob)
        training_outputs = [tf.matmul(dropped_outputs[idx], out_weights) + out_biases for idx in range(n_steps)]

        predicted_outputs = [tf.matmul(hidden_outputs[idx], out_weights) + out_biases for idx in range(n_steps)]
        final_output = tf.transpose(predicted_outputs, perm=[1, 0, 2])

        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=training_outputs, labels=y_label))
        l2_loss = tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(out_weights)
        loss = tf.reduce_mean(ce_loss + (l2_beta * l2_loss))
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=10)


class TaggerModel:
    def __init__(self, emb, model_path=None):
        self.embedding = emb

        self.session = tf.Session(graph=graph)
        if model_path:
            saver.restore(self.session, model_path)
        else:
            self.session.run(init)

    def train(self, train_input, train_label, train_sent_len):
        feed_dict = {
            x: train_input,
            y: train_label,
            embedding: self.embedding,
            sentence_len: train_sent_len
        }
        _, cross_entropy_loss = self.session.run([optimizer, loss], feed_dict=feed_dict)
        return cross_entropy_loss

    def evaluate(self, eval_input, eval_label, eval_sent_len):
        feed_dict = {
            x: eval_input,
            y: eval_label,
            embedding: self.embedding,
            sentence_len: eval_sent_len
        }

        cross_entropy_loss, predicted_result = self.session.run([loss, final_output], feed_dict=feed_dict)
        return predicted_result, cross_entropy_loss

    def predict(self, test_input, test_sent_len):
        feed_dict = {
            x: test_input,
            embedding: self.embedding,
            sentence_len: test_sent_len
        }

        predicted_result = self.session.run(final_output, feed_dict=feed_dict)
        return predicted_result

    def save_model(self, model_path, global_step):
        saver.save(self.session, model_path, global_step=global_step)
