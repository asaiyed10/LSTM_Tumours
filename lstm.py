import tensorflow as tf
from tensorflow.contrib import rnn


class LSTM:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batchs', 1)
        self.input_size = kwargs.get('inputs', 1024)
        self.step_size = kwargs.get('steps', 128)
        self.output_size = kwargs.get('outputs', 2)
        self.learn_rate = kwargs.get('alpha', 0.001)
        self.ae = kwargs['ae']

        self.inputs = tf.placeholder(tf.float32,
                                     [None, self.step_size, self.input_size])
        self.outputs = tf.placeholder(tf.float32, [None, self.output_size])

        self.hidden = kwargs.get('hidden', 256)
        self.weights = tf.Variable(
            tf.random_normal([self.hidden, self.output_size]))
        self.bias = tf.Variable(tf.random_normal([self.output_size]))

        self.network = self.__lstm()
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.network,
                                                    labels=self.outputs))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learn_rate).minimize(self.loss)

        correct = tf.equal(tf.argmax(self.network, 1),
                           tf.argmax(self.outputs, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100

    def __lstm(self):
        x = tf.unstack(self.inputs, self.step_size, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(self.hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], self.weights) + self.bias

    def train(self, sess, patterns, iterations, show_on):
        for i in range(iterations):
            batch_x, batch_y = self.ae.predict(sess, patterns, self.batch_size)
            feed_dict = {self.inputs: batch_x, self.outputs: batch_y}

            sess.run(self.optimizer, feed_dict=feed_dict)
            if i % show_on == 0:
                acc = sess.run(self.accuracy, feed_dict=feed_dict)
                loss = sess.run(self.loss, feed_dict=feed_dict)
                print('Iteration ' + str(i).rjust(5, ' ') +
                      '\tLoss: ' + str(round(loss, 2)).rjust(5, ' ') +
                      '\tAccuracy: ' + str(round(acc, 2)).rjust(5, ' '))

    def predict(self, sess, patterns):
        batch_x, batch_y = self.ae.predict(sess, patterns, len(patterns))
        feed_dict = {self.inputs: batch_x, self.outputs: batch_y}

        acc = sess.run(self.accuracy, feed_dict=feed_dict)
        return str(round(acc, 2)).rjust(5, ' ')
