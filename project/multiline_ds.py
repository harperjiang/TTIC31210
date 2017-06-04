import numpy as np

from ndnn.dataset import Batch


class MultilineDataset:
    def __init__(self, file, num_line, extds = None):
        if extds == None:
            self.enc_dict = {}
            self.enc_word = []
            self.dec_dict = {}
            self.dec_word = []
        else:
            self.enc_dict = extds.enc_dict
            self.enc_word = extds.enc_word
            self.dec_dict = extds.dec_dict
            self.dec_word = extds.dec_word

        f = open(file, "r")

        self.records = []
        buffer = []
        head = None
        counter = 0
        for line in f.readlines():
            if counter == 0:
                if len(buffer) > 0:
                    self.records.append((head, buffer))
                buffer = []
                head = self.decParse(line)
            else:
                buffer.append(self.encParse(line))
            counter = (counter + 1) % (num_line + 1)
        self.records.append((head, buffer))

    def encParse(self, line):
        words = line.split()
        for word in words:
            if word not in self.enc_dict:
                self.enc_dict[word] = len(self.enc_dict)
                self.enc_word.append(word)
        return [self.enc_dict[w] for w in words]

    def decParse(self, line):
        words = line.split()
        for word in words:
            if word not in self.dec_dict:
                self.dec_dict[word] = len(self.dec_dict)
                self.dec_word.append(word)
        return [self.dec_dict[w] for w in words]

    # Ignore batch size
    def batches(self, batch_size):
        for record in self.records:
            yield Batch(1, np.int32(record[1]), np.int32(record[0]))
