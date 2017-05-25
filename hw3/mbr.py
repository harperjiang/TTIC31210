import numpy as np

from dataset import UDDataSet
from gibbs import Gibbs
from hmm import HMM

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu', train_ds)

hmm = HMM(train_ds)
gibbs = Gibbs(hmm)


def predict(iteration):
    num_total_tag = 0
    num_correct_tag = 0

    for sentence in dev_ds.sentences():
        gibbs.sample(sentence, iteration)

        data = np.array(gibbs.states)
        k, r = data.shape
        predict_tag = np.zeros(r)
        # Compute Predict for each position
        for i in range(r):
            predict_tag[i] = np.argmax(np.bincount(data[:, i]))
        gt_tag = np.array([word[1] for word in sentence.words])
        num_correct_tag += (predict_tag == gt_tag).sum()
        num_total_tag += r
    return num_correct_tag / num_total_tag


k = [1, 2, 5, 10, 100, 500, 1000, 2000]

# gibbs.beta_schedule = BetaSchedule()

for ite in k:
    print("%d & %.4f\\\\\\hline" % (ite, predict(ite)))
