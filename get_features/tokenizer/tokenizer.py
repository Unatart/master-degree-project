import numpy as np


def vocab_creater(array):
    word2idx = {}
    idx2word = {}
    for i in range(len(array)):
        word2idx[array[i]] = i
        idx2word[i] = array[i]

    return word2idx, idx2word, len(array)


def create_numerate_array(array):
    word2idx, idx2word, _ = vocab_creater(np.unique(array))
    new = []

    for i in range(len(array)):
        new.append(word2idx[array[i]])

    return new
