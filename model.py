import tensorflow as tf
import argparse as ag
from scans import CTScan
from ae import AutoEncoder
from lstm import LSTM

# Data Parameters.
batch_size = 2
pat_size = 512
scan_size = 128
pool_size = 16

# Network Parameters.
code_size = 32
lstm_in = code_size**2
classes = 2
hidden_size = 256

# Training Parameters.
learning_rate = 0.001
iterations = 10
show_on = 1


def parse_args():
    parser = ag.ArgumentParser(description='Train an lstm on lung tumor scans.')
    parser.add_argument('-t', help='Folder containing training scans.',
                        type=str, default='train_images')
    parser.add_argument('-ts', help='Folder containing testing scans.',
                        type=str, default='test_images')
    parser.add_argument('-l', help='File containing scan labels.',
                        type=str, default='stage1_labels.csv')
    parser.add_argument('-a', help='Retrain autoencoder', action='store_true')
    parser.add_argument('-s', help='Save session to file.',
                        type=str, default=None)
    parser.add_argument('-r', help='Load session to file.',
                        type=str, default=None)
    return parser.parse_args()

args = parse_args()
train_folder = args.t
test_folder = args.ts
label_file = args.l

train_scans = CTScan(pat_dir=train_folder, label_file=label_file)
test_scans = CTScan(pat_dir=test_folder, label_file=label_file)

ae = AutoEncoder(batchs=batch_size, inputs=pat_size, steps=scan_size,
                 alpha=learning_rate, pools=pool_size, hidden=code_size)
lstm = LSTM(ae=ae, batchs=batch_size, inputs=lstm_in, steps=scan_size,
            outputs=classes, alpha=learning_rate, hidden=hidden_size)

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    if args.r is not None:
        saver.restore(sess, args.r)
    else:
        sess.run(tf.global_variables_initializer())
    print('Network initialized.')

    if args.a:
        print('Training autoencoder.')
        ae.train(sess, train_scans, iterations, show_on)
        if args.s is not None:
            print('Autoencoder trained, saving autoencoder.')
            saver.save(sess, args.s)
            print('Autoencoder saved, training LSTM.')
        else:
            print('Autoencoder trained, training LSTM.')
    else:
        print('Training LSTM.')

    lstm.train(sess, train_scans, iterations, show_on)
    print('Train accuracy: ' + lstm.predict(sess, train_scans))

    if args.s is not None:
        print('LSTM trained, saving network.')
        saver.save(sess, args.s)
        print('Network saved, predicting test dataset.')
    else:
        print('LSTM trained, predicting test dataset.')

    print('Test accuracy: ' + lstm.predict(sess, test_scans))
