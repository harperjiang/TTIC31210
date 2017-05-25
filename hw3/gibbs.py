import math

import numpy as np


class Gibbs:
    def __init__(self, hmm):
        self.hmm = hmm
        self.beta = 1
        self.beta_schedule = None

    def sample(self, sentence, iteration):
        if self.beta_schedule is not None:
            self.beta_schedule.reset()

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
            # Update Beta
            if self.beta_schedule is not None:
                self.beta = self.beta_schedule.update

        return state

    @staticmethod
    def pick(dist, probe):
        cum_sum = -np.inf
        for i in range(len(dist)):
            cum_sum = np.logaddexp(cum_sum, dist[i])
            if cum_sum >= probe:
                return i
        raise Exception()


class BetaSchedule:
    def __init__(self):
        self.beta = 0.1

    def reset(self):
        self.beta = 0.1

    def update(self):
        self.beta += 0.1
        return self.beta
