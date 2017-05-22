'''
Created on May 11, 2017

@author: harper
'''


def get_dict():
    lines = open("bobsue-data/bobsue.voc.txt", "rb").readlines()
    vocab_dict = {}
    idx_dict = []
    for line in lines:
        word = line.decode('utf-8', errors='replace').strip()
        vocab_dict[word] = len(vocab_dict)
        idx_dict.append(word)
    return vocab_dict, idx_dict


def translate(idx_dict, data):
    # Remove everything after end symbol
    if isinstance(data,list) and 1 in data:
        del data[data.index(1)+1:]

    words = [idx_dict[i] for i in data]
    return " ".join(words)
