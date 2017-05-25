import unittest

from dataset import UDDataSet, UNK


class UDDataSetTest(unittest.TestCase):
    def test(self):
        train_ds = UDDataSet('data/en-ud-train.conllu')
        test_ds = UDDataSet('data/en-ud-dev.conllu', train_ds)

        self.assertEqual(UNK, train_ds.lookup_word('UNKDSF'))
        self.assertEqual(train_ds.pos, test_ds.pos)
        self.assertEqual(train_ds.words, test_ds.words)

        self.assertEqual(6, train_ds.lookup_pos("ADP"))

    def test_find_word(self):
        train_ds = UDDataSet('data/en-ud-train.conllu')
        print(train_ds.idx2word(121))
        print(train_ds.lookup_pos("PART"))
