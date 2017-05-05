import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# parameters
embedding_dim = 64
batch_size = 128
learning_rate = 0.001
vocabulary_size = 100004

# network parameters
n_steps = 30
n_input = 64
n_hidden = 100
n_classes = 60
n_bilstm_stack = 2

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.int32, [batch_size, n_steps])
    y = tf.placeholder(tf.int32, [batch_size, n_steps])
    sentence_len = tf.placeholder(tf.int32, [batch_size])
    embedding = tf.placeholder("float", [vocabulary_size, n_input])

    with tf.device('/cpu:0'):
        out_weights = tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
        out_biases = tf.Variable(tf.random_normal([n_classes]))

        x_input = tf.nn.embedding_lookup(embedding, x)
        x_input = tf.transpose(x_input, [1, 0, 2])
        x_input = tf.reshape(x_input, [-1, n_input])
        x_input = tf.split(x_input, n_steps, 0)

        y_label = tf.one_hot(y, depth=n_classes, on_value=1.0, off_value=0.0, axis=-1)
        y_label = tf.transpose(y_label, [1, 0, 2])
        y_label = tf.reshape(y_label, [-1, n_classes])
        y_label = tf.split(y_label, n_steps, 0)

        lstm_fw_cells = [rnn.BasicLSTMCell(n_hidden)] * n_bilstm_stack
        lstm_bw_cells = [rnn.BasicLSTMCell(n_hidden)] * n_bilstm_stack

        outputs1, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cells[0], lstm_bw_cells[0], x_input,
                                                     dtype=tf.float32, sequence_length=sentence_len)


        # TODO: Edit Tagger model
        predicted = [tf.matmul(outputs1[idx], out_weights) + out_biases for idx in range(n_steps)]

        final_output = tf.transpose(predicted, perm=[1, 0, 2])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=y_label))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


class TaggerModel:
    def __init__(self, emb, model_path=None):
        self.embedding = emb

        self.session = tf.Session(graph=graph)
        if model_path:
            saver.restore(self.session, model_path)

        self.session.run(init)

    def train(self, train_input, train_label, train_sent_len):
        feed_dict = {
            x: train_input,
            y: train_label,
            embedding: self.embedding,
            sentence_len: train_sent_len
        }
        # _, cross_entropy_loss = self.session.run([optimizer, loss], feed_dict=feed_dict)
        out = self.session.run(outputs, feed_dict=feed_dict)
        print(np.asarray(out).shape)
        # return cross_entropy_loss
        return 0

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