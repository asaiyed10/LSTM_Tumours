import tensorflow as tf
import argparse as ag
from scans import CTScan

fold_size = 7


def parse_args():
    parser = ag.ArgumentParser(description='Train an lstm on lung tumor scans.')
    parser.add_argument('-t', help='Folder containing training scans.',
                        type=str, default='train_images')
    parser.add_argument('-ts', help='Folder containing testing scans.',
                        type=str, default='test_images')
    parser.add_argument('-l', help='File containing scan labels.',
                        type=str, default='stage1_labels.csv')
    return parser.parse_args()

args = parse_args()
train_folder = args.t
test_folder = args.ts
label_file = args.l

train_scans = CTScan(pat_dir=train_folder, label_file=label_file)
test_scans = CTScan(pat_dir=test_folder, label_file=label_file)

# print('\n\n'.join('\n'.join(str(y) for y in x.patterns)
#                   for x in train_scans.fold(fold_size)))

# print('\n'.join(str(len(x)) for x in train_scans.fold(fold_size)))
print(len(train_scans.fold(fold_size)))
print(sum(len(x.patterns) for x in train_scans.fold(fold_size)))
