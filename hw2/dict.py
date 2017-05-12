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

