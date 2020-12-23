# -*- coding:UTF-8 -*-
import itertools
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import *

plt.switch_backend('agg')


def zero_padding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binary_matrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(False)
            else:
                m[i].append(True)
    return m


# Returns padded input sequence tensor and lengths
def input_var(indexes_batch):
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def output_var(indexes_batch):
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    mask = binary_matrix(padList)
    mask = torch.tensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2data(pair_batch, sort=False):
    if not sort:
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch)
    output, mask, max_target_len = output_var(output_batch)
    return inp, lengths, output, mask, max_target_len


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 该定时器用于定时记录时间
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(fig_path, dpi=300)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
