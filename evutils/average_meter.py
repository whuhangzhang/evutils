# -*- coding:utf-8 -*-
# ref: https://github.com/donnyyou/torch-segmentation/blob/75a3932b1fa676c022cc42f2ad556dc6062e79cf/lib/tools/util/average_meter.py
import time


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictAverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.key_list = None

    def reset(self):
        self.val = {key:0. for key in self.key_list}
        self.avg = {key:0. for key in self.key_list}
        self.sum = {key:0. for key in self.key_list}
        self.count = {key:0 for key in self.key_list}

    def update(self, val_dict, n_dict=None):
        if self.key_list is None:
            self.key_list = val_dict.keys()
            self.reset()

        if isinstance(n_dict, (int, float)):
            new_n_dict = {k: n_dict for k in val_dict.keys()}
            n_dict = new_n_dict

        self.val = val_dict
        for k in val_dict.keys():
            self.sum[k] += val_dict[k] * n_dict[k]
            self.count[k] += n_dict[k]
            self.avg[k] = self.sum[k] / self.count[k]

    def info(self):
        str = '{'
        for k, v in self.avg.items():
            str += '{}: {:.4f}, '.format(k, v)

        return str.rstrip(', ') + '}'


class ListAverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.len = None

    def reset(self):
        self.val = [0.] * self.len
        self.avg = [0.] * self.len
        self.sum = [0.] * self.len
        self.count = [0] * self.len

    def update(self, val_list, n_list=None):
        if self.len is None:
            self.len = len(val_list)
            self.reset()

        if isinstance(n_list, (int, float)):
            new_n_list = [n_list] * len(val_list)
            n_list = new_n_list

        self.val = val_list
        for k in range(len(val_list)):
            self.sum[k] += val_list[k] * n_list[k]
            self.count[k] += n_list[k]
            self.avg[k] = self.sum[k] / self.count[k]

    def info(self):
        return "[" + ', '.join(['{:.4f}'.format(x) for x in self.avg]) + "]"


class TimeAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._cnt = 0
        self._total_time = 0
        self._total_samples = 0

    def record(self, usetime, num_samples=None):
        self._cnt += 1
        self._total_time += usetime
        if num_samples:
            self._total_samples += num_samples

    def get_average(self):
        if self._cnt == 0:
            return 0
        return self._total_time / float(self._cnt)

    def get_ips_average(self):
        if not self._total_samples or self._cnt == 0:
            return 0
        return float(self._total_samples) / self._total_time
