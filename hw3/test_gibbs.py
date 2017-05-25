import unittest

from dataset import UDDataSet
from gibbs import Gibbs
from hmm import HMM


class GibbsTest(unittest.TestCase):
    def test_sample(self):
        train_ds = UDDataSet("data/en-ud-train.conllu")
        dev_ds = UDDataSet("data/en-ud-dev.conllu", train_ds)

        gibbs = Gibbs(HMM(train_ds))

        gibbs.sample(dev_ds.sentences()[1], 10)
