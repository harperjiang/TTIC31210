from dataset import UDDataSet
from gibbs import Gibbs
from hmm import HMM

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu', train_ds)

hmm = HMM(train_ds)

gibbs = Gibbs(hmm)

for sentence in dev_ds.sentences():
    print([train_ds.idx2pos(x) for x in gibbs.sample(sentence, 10)])
