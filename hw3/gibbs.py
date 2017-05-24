import numpy as np


class Gibbs:
    def __init__(self, hmm):
        self.hmm = hmm

    def sample(self, sentence, iteration):
        state = ["BOS"] + [np.random.randint(0, self.hmm.num_state) for i in range(len(sentence))] + ["EOS"]
        for idx in range(iteration):
            for sidx in range(len(sentence)):
                xt = sentence.words[sidx][0]
                if sidx == 0:  # Start
                    ytp1 = sentence.words[sidx + 1][1]
                    ytm1 = None
                elif sidx == len(sentence) - 1:  # End
                    ytm1 = sentence.words[sidx - 1][1]
                    ytp1 = None
                else:
                    ytp1 = sentence.words[sidx + 1][1]
                    ytm1 = sentence.words[sidx - 1][1]
                # HMM.cond_prob returns a conditional probability
                prob_dist = self.hmm.cond_prob(xt, ytm1, ytp1)
