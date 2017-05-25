import numpy as np

from dataset import UDDataSet
from gibbs import Gibbs, BetaSqSchedule
from hmm import HMM

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu', train_ds)

hmm = HMM(train_ds)
gibbs = Gibbs(hmm)


def predict(iteration):
    num_total_tag = 0
    num_correct_tag = 0

    for sentence in dev_ds.sentences():
        predict_tag = gibbs.sample(sentence, iteration)
        num_total_tag += len(sentence)
        gt_tag = np.array([word[1] for word in sentence.words])
        num_correct_tag += (predict_tag == gt_tag).sum()

    # print("Tag accuracy: %.4f" % (num_correct_tag / num_total_tag))
    return num_correct_tag / num_total_tag


k = [1, 2, 5, 10, 100, 500, 1000, 2000]

gibbs.beta_schedule = BetaSqSchedule()

for ite in k:
    print("%d & %.4f\\\\\\hline" % (ite, predict(ite)))
