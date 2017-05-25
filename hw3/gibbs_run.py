import numpy as np

from dataset import UDDataSet
from gibbs import Gibbs
from hmm import HMM

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu', train_ds)

hmm = HMM(train_ds)

gibbs = Gibbs(hmm)

iteration = 10

num_total_tag = 0
num_correct_tag = 0

for idx,sentence in enumerate(dev_ds.sentences()):
    try:
        predict_tag = gibbs.sample(sentence, iteration)
    except Exception:
        print(idx)
        raise Exception()
    num_total_tag += len(sentence)
    gt_tag = np.array([word[1] for word in sentence.words])
    num_correct_tag += (predict_tag == gt_tag).sum()

print("Tag accuracy: %.4f" % (num_correct_tag / num_total_tag))
