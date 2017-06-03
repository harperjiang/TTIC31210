import numpy as np

from ndnn.dataset import Batch


class S2SDict:
    def __init__(self, files):
        self.enc_dict = {}
        self.enc_inv_dict = []
        self.dec_dict = {}
        self.dec_inv_dict = []

        self.enc_dict["<s>"] = 0
        self.enc_dict["</s>"] = 1
        self.enc_inv_dict.append("<s>")
        self.enc_inv_dict.append("</s>")

        self.dec_dict["<s>"] = 0
        self.dec_dict["</s>"] = 1
        self.dec_inv_dict.append("<s>")
        self.dec_inv_dict.append("</s>")

        for file in files:
            f = open(file, "r")
            for line in f.readlines():
                sents = line.split("\t")
                sent0 = sents[0].split()
                sent1 = sents[1].split()

                for word in sent0:
                    if word not in self.enc_dict:
                        self.enc_dict[word] = len(self.enc_dict)
                        self.enc_inv_dict.append(word)

                for word in sent1:
                    if word not in self.dec_dict:
                        self.dec_dict[word] = len(self.dec_dict)
                        self.dec_inv_dict.append(word)
            f.close()

    def enc_translate(self, input):
        pass

    def dec_translate(self, input):
        pass


class LSTMDataSet:
    def __init__(self, vocab_dict, idx_dict, filename):
        self.vocab_dict = vocab_dict
        self.idx_dict = idx_dict
        datas = {}
        lines = open(filename, "rb").readlines()

        for line in lines:
            words = line.decode('utf-8', errors='replace').split()

            idx = np.ndarray((len(words),), dtype=np.int32)
            for i, word in enumerate(words):
                if word not in self.vocab_dict:
                    raise Exception()
                idx[i] = self.vocab_dict[word]
            keylen = len(idx)
            if keylen not in datas:
                datas[keylen] = []
            datas[keylen].append(idx)

        self.datas = list(datas.values())

    def translate_to_str(self, numarray):
        return ' '.join([self.idx_dict[n] for n in numarray])

    def num_batch(self):
        return self.numbatch

    def batches(self, batch_size):
        self.numbatch = np.sum([np.ceil(len(item) / batch_size) for item in self.datas])

        perm = np.random.permutation(len(self.datas))

        for p in perm:
            batch_base = self.datas[p]
            subperm = np.random.permutation(len(batch_base))
            batch_range = range(0, len(batch_base), batch_size)

            batch_idices = [subperm[idx:idx + batch_size] for idx in batch_range]

            for batch_idx in batch_idices:
                item = [batch_base[idx] for idx in batch_idx]
                yield Batch(len(item), np.array(item), None)


class S2SDataSet:
    def __init__(self, enc_dict, dec_dict, filename):
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        datas = {}
        lines = open(filename, "rb").readlines()

        for line in lines:
            sentences = line.decode('utf-8', errors='replace').split('\t')

            psent = sentences[0]
            csent = sentences[1]

            pwordIdx = [self.enc_dict[word] for word in psent.split()]
            cwordIdx = [self.dec_dict[word] for word in csent.split()]

            plen = len(pwordIdx)
            clen = len(cwordIdx)

            key = (plen, clen)
            if key not in datas:
                datas[key] = []
            datas[key].append([pwordIdx, cwordIdx])

        self.datas = list(datas.values())
        self.numbatch = 0

    def num_batch(self):
        return self.numbatch

    def batches(self, batch_size):
        self.numbatch = np.sum([np.ceil(len(item) / batch_size) for item in self.datas])

        perm = np.random.permutation(len(self.datas))

        for p in perm:
            batch_base = self.datas[p]
            subperm = np.random.permutation(len(batch_base))
            batch_range = range(0, len(batch_base), batch_size)

            batch_idices = [subperm[idx:idx + batch_size] for idx in batch_range]

            for batch_idx in batch_idices:
                item1 = []
                item2 = []
                for idx in batch_idx:
                    item1.append(batch_base[idx][0])
                    item2.append(batch_base[idx][1])
                yield Batch(len(batch_idx), [np.array(item1), np.array(item2)], None)
