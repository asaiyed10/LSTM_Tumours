import argparse as ag
from scans import CTScan
from pympler import tracker

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

tr = tracker.SummaryTracker()
tr.print_diff()


# Launch the graph
