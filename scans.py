import numpy as np
import os
import dicom
import pandas
from random import sample, shuffle
from pympler import muppy
from pympler import summary

all_objects = muppy.get_objects()
print(len(all_objects))

sum1 = summary.summarize(all_objects)
summary.print_(sum1)

sum2 = summary.summarize(muppy.get_objects())
diff = summary.get_diff(sum1, sum2)
summary.print_(diff)
` e1
class CTScan:
    def __init__(self, **kwargs):
        pat_dir = kwargs.get('pat_dir', None)
        patterns = kwargs.get('patterns', None)
        if pat_dir is not None:
            self.patterns = list(os.walk(pat_dir))[1:]
        else:
            self.patterns = patterns

        label_file = kwargs.get('label_file', None)
        labels = kwargs.get('labels', None)
        if label_file is not None:
            self.labels = pandas.read_csv(label_file)
        else:
            self.labels = labels

    def __len__(self):
        return len(self.patterns)

    def scans(self):
        for p in self.patterns:
            yield (p[0][p[0].index('/') + 1:],
                   [dicom.read_file(p[0] + '/' + file).pixel_array
                    for file in p[2]])

    def fold(self, num_folds):
        tmp_pats = self.patterns

        fold_size = int(len(self.patterns) / num_folds)
        spare = len(self.patterns) % num_folds

        fold_scans = []
        for i in range(num_folds):
            shuffle(tmp_pats)

            width = fold_size + 1 if i < spare else fold_size
            fold_pat = tmp_pats[0:width]
            tmp_pats = tmp_pats[width:]

            fold_scans += [CTScan(patterns=fold_pat, labels=self.labels)]

        return fold_scans

    @staticmethod
    def reduce(pattern, size):
        pattern = np.array(pattern)
        distance = len(pattern) - size
        if distance > 0:
            to_remove = sample(range(len(pattern)), distance)
            return np.delete(pattern, to_remove, axis=0)
        elif distance < 0:
            to_add = [np.zeros((512, 512)) for _ in range(abs(distance))]
            return np.append(pattern, to_add, axis=0)
        return pattern

    def next_batch(self, batch_size, scan_width=120):
        batch_x, batch_y = [], []
        pats = self.scans()
        labels = {v: k for k, v in dict(self.labels.id).items()}

        if batch_size > len(self.patterns):
            raise AttributeError('Cannot have batch larger than dataset.')

        for i in range(batch_size):
            pat_id, pattern = next(pats)
            pattern = self.reduce(pattern, scan_width)
            label = 0 if pat_id not in labels.keys() else \
                self.labels.cancer[labels[pat_id]]
            label = [1-label, 1-label]

            batch_x += [pattern]
            batch_y += [label]


        return np.array(batch_x), np.array(batch_y)
