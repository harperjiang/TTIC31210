import unittest

import numpy as np

from dataset import UDDataSet
from hmm import HMM


class HMMTest(unittest.TestCase):
    def test_init(self):
        hmm = HMM(UDDataSet('data/en-ud-train.conllu'))

        self.assertEqual(17, hmm.num_state)
        self.assertEqual(17, hmm.bos_idx)
        self.assertEqual(18, hmm.eos_idx)

    def test_cond_prop(self):
        hmm = HMM(UDDataSet("data/en-ud-train.conllu"))

        cprob = hmm.cond_prob(4, 3, 7)
        self.assertAlmostEqual(1.00, sum([np.exp(i) for i in cprob]))

        often = hmm.dataset.lookup_word("often")
        adj = hmm.dataset.lookup_pos("ADJ")
        propn = hmm.dataset.lookup_pos("PROPN")
        cprob = hmm.cond_prob(often, adj, propn)
        self.assertAlmostEqual(1.00, sum([np.exp(i) for i in cprob]))

    def test_normalize(self):
        hmm = HMM(UDDataSet("data/en-ud-train.conllu"))
        counter = {}
        counter['a'] = 1
        counter['b'] = 2
        counter['c'] = 3
        counter['d'] = 4
        hmm.normalize(counter)

        self.assertEqual(np.log(0.1),counter['a'])
        self.assertEqual(np.log(0.2),counter['b'])
        self.assertEqual(np.log(0.3),counter['c'])
        self.assertEqual(np.log(0.4),counter['d'])

