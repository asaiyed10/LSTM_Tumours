import tensorflow as tf
import numpy as np


class AutoEncoder:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batchs', 1)
        self.input_size = kwargs.get('inputs', 512)
        self.step_size = kwargs.get('steps', 128)
        self.learn_rate = kwargs.get('alpha', 0.001)

        # Defining network dimensions.
        self.pool_size = kwargs.get('pools', 32)
        self.pool_dim = int(self.input_size/self.pool_size)**2
        self.hidden_size = kwargs.get('hidden', 20)**2
        self.output_size = self.pool_dim

        self.inputs = tf.placeholder(tf.float32)
        self.outputs = tf.placeholder(tf.float32)

        # Pool data to avoid large neural networks.
        self.pool = tf.nn.max_pool(
            self.inputs, ksize=[1, self.pool_size, self.pool_size, 1],
            strides=[1, self.pool_size, self.pool_size, 1],
            padding='SAME')
        self.reshape = tf.reshape(self.pool, [-1, self.pool_dim])

        # Hidden layer
        self.hidden_w = tf.Variable(tf.truncated_normal(
            [self.pool_dim, self.hidden_size]))
        self.hidden_b = tf.Variable(tf.truncated_normal(
            [self.hidden_size]))
        self.h1 = tf.nn.relu(
            tf.matmul(self.reshape, self.hidden_w) + self.hidden_b)

        # Visible layer.
        self.vis_w = tf.Variable(tf.truncated_normal(
            [self.hidden_size, self.output_size]))
        self.vis_b = tf.Variable(tf.truncated_normal(
            [self.output_size]))
        self.y = tf.nn.relu(tf.matmul(self.h1, self.vis_w) + self.vis_b)
        self.xbar = tf.reshape(self.y, [-1, self.pool_dim])

        # Loss & optimizer.
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.xbar - self.reshape))
        self.optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(
            self.loss)

        # Calculating accuracy.
        correct = tf.equal(tf.argmax(self.xbar, 1),
                           tf.argmax(self.outputs, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def train(self, sess, patterns, iterations, show_on):
        for i in range(iterations):
            batch_x, _ = patterns.next_batch(self.batch_size, self.step_size)

            for j, pattern in enumerate(batch_x):
                pattern = np.reshape(pattern, [-1, self.input_size,
                                               self.input_size, 1])
                feed_dict = {self.inputs: pattern, self.outputs: pattern}
                _, instant_loss, logits, label = sess.run(
                    [self.optimizer, self.loss, self.xbar, self.reshape],
                    feed_dict=feed_dict)

                if j == 0:
                    pred = np.array([logits])
                    labels = np.array([label])
                else:
                    pred = np.append(pred, [logits], axis=0)
                    labels = np.append(labels, [label], axis=0)

            if i % show_on == 0:
                print('Training Ae: ' + str(i).rjust(3, ' ') +
                      ' of ' + str(iterations))

    def predict(self, sess, patterns, batch):
        pred = []
        batch_x, batch_y = patterns.next_batch(batch, self.step_size)

        for i, pattern in enumerate(batch_x):
            pattern = np.reshape(pattern, [-1, self.input_size,
                                           self.input_size, 1])
            feed_dict = {self.inputs: pattern}
            logits = sess.run([self.h1], feed_dict=feed_dict)

            if i == 0:
                pred = np.array(logits)
            else:
                pred = np.append(pred, logits, axis=0)

        return pred, batch_y
