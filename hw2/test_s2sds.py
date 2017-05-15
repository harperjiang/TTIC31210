import unittest

from lstm_dataset import S2SDataSet
from vocab_dict import get_dict


class S2SDataSetTest(unittest.TestCase):
    def testLoad(self):
        vdict, idict = get_dict()

        ds = S2SDataSet(vdict, idict, 'bobsue-data/bobsue.seq2seq.dev.tsv')

        for batch in ds.batches(30):
            self.assertEqual(2, len(batch.data))
            self.assertEqual(batch.data[0].shape[0],batch.data[1].shape[0])
