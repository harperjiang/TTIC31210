import math

import numpy as np


class Gibbs:
    def __init__(self, hmm):
        self.hmm = hmm
        self.beta = 1

    def sample(self, sentence, iteration):
        num_state = self.hmm.num_state
        sent_len = len(sentence)
        state = np.random.randint(0, num_state, size=sent_len)
        for idx in range(iteration):
            for sidx in range(sent_len):
                xt = sentence.words[sidx][0]
                if sidx == 0:  # Start
                    if sent_len == 1:
                        ytp1 = None
                    else:
                        ytp1 = state[sidx + 1]
                    ytm1 = None
                elif sidx == len(sentence) - 1:  # End
                    ytm1 = state[sidx - 1]
                    ytp1 = None
                else:
                    ytp1 = state[sidx + 1]
                    ytm1 = state[sidx - 1]
                # HMM.cond_prob returns a conditional probability
                prob_dist = self.hmm.cond_prob(xt, ytm1, ytp1, self.beta)
                # Debug for nan
                if math.isnan(prob_dist[0]):
                    raise Exception("%d,%d,%d" % (xt, ytm1, ytp1))
                # Sample from the prob
                probe = np.log(np.random.rand())
                state[sidx] = self.pick(prob_dist, probe)
        return state

    @staticmethod
    def pick(dist, probe):
        cum_sum = -np.inf
        for i in range(len(dist)):
            cum_sum = np.logaddexp(cum_sum, dist[i])
            if cum_sum >= probe:
                return i
        raise Exception()

class AnnealingSchedule:
    def __init__(self):
        pass

    def update(self, beta):
        pass
