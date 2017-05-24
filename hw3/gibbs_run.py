from dataset import UDDataSet
from hmm import HMM

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu')

hmm = HMM()
hmm.train(train_ds)

