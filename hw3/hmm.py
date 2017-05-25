import numpy as np

from dataset import UNK


class HMM:
    def __init__(self, train_ds):
        self.emission_counter = {}
        self.transition_counter = {}
        self.dataset = train_ds

        # Train HMM by doing counting
        self.num_state = len(self.dataset.pos)
        self.bos_idx = self.num_state
        self.eos_idx = self.num_state + 1

        for sent in self.dataset.sentences():
            # Count transition from BOS
            prev_word = (-1, self.bos_idx)
            for word in sent.words:
                emit_key = (word[1], word[0])
                if emit_key not in self.emission_counter:
                    self.emission_counter[emit_key] = 0
                self.emission_counter[emit_key] += 1
                transit_key = (prev_word[1], word[1])
                if transit_key not in self.transition_counter:
                    self.transition_counter[transit_key] = 0
                self.transition_counter[transit_key] += 1
                prev_word = word

            # Count transition to EOS
            transit_key = (prev_word[1], self.eos_idx)
            if transit_key not in self.transition_counter:
                self.transition_counter[transit_key] = 0
            self.transition_counter[transit_key] += 1

        # Smoothing
        unk_idx = self.dataset.lookup_word(UNK)
        for pos in self.dataset.idxpos:
            self.emission_counter[(pos, unk_idx)] = 1

        self.normalize(self.emission_counter)
        self.normalize(self.transition_counter)

    # Compute the conditional probability p(y_t | others)
    # using p(y_t | y_{t-1}) p(x_t | y_t) p(y_{t+1}|y_t)
    def cond_prob(self, xt, ytm1, ytp1):
        if ytm1 is None:
            ytm1 = self.bos_idx
        if ytp1 is None:
            ytp1 = self.eos_idx
        # Compute
        result = np.zeros(self.num_state)
        for i in range(self.num_state):
            ytytm1 = self.transition_counter.get((ytm1, i), -np.inf)
            xtyt = self.emission_counter.get((i, xt), -np.inf)
            ytp1yt = self.transition_counter.get((i, ytp1), -np.inf)
            result[i] = ytytm1 + xtyt + ytp1yt

        # Normalize log value
        cum_sum = -np.inf
        for i in result:
            cum_sum = np.logaddexp(cum_sum, i)
        for i in range(self.num_state):
            result[i] = result[i] - cum_sum

        return result

    # Normalize
    @staticmethod
    def normalize(counter):
        sum_val = sum(v for v in counter.values())
        for key in counter:
            counter[key] /= sum_val
            counter[key] = np.log(counter[key])
