import numpy as np


class Gibbs:
    def __init__(self, hmm):
        self.hmm = hmm

    def sample(self, sentence, iteration):
        num_state = self.hmm.num_state
        state = [np.random.randint(0, num_state) for i in range(len(sentence))]
        for idx in range(iteration):
            sent_len = len(sentence)
            for sidx in range(sent_len):
                xt = sentence.words[sidx][0]
                if sidx == 0:  # Start
                    if sent_len == 1:
                        ytp1 = None
                    else:
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
                # Sample from the prob
                probe = np.log(np.random.rand())
                cum_sum = -np.inf
                pick = -1
                for i in range(len(prob_dist)):
                    cum_sum = np.logaddexp(cum_sum, prob_dist[i])
                    if cum_sum >= probe:
                        pick = i
                        break
                state[sidx] = pick
        return state
