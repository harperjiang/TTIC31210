from dataset import UNK, BOS, EOS


# Normalize
def normalize(counter):
    sum_val = sum(v for v in counter.values())
    for key in counter:
        counter[key] /= sum_val


class HMM:
    def __init__(self):
        self.emission_counter = {}
        self.transition_counter = {}
        self.num_state = 0
        self.dataset = None

    def train(self, train_ds):
        self.dataset = train_ds
        # Train HMM by doing counting

        self.emission_counter.clear()
        self.transition_counter.clear()

        for sent in self.dataset.sentences():
            bos_idx = self.dataset.lookup_pos(BOS)
            eos_idx = self.dataset.lookup_pos(EOS)
            prev_word = (-1, bos_idx)
            for word in (sent.words() + [(-1, eos_idx)]):
                emit_key = (word[1], word[0])
                if emit_key not in self.emission_counter:
                    self.emission_counter[emit_key] = 0
                self.emission_counter[emit_key] += 1
                if prev_word is not None:
                    transit_key = (prev_word[1], word[1])
                    if transit_key not in self.transition_counter:
                        self.transition_counter[transit_key] = 0
                    self.transition_counter[transit_key] += 1
                prev_word = word

        # Smoothing
        unk_idx = self.dataset.lookup_word(UNK)
        for pos in self.dataset.idxpos:
            self.emission_counter[(pos, unk_idx)] = 1

        normalize(self.emission_counter)
        normalize(self.transition_counter)
        self.num_state = len(train_ds.pos)

    def cond_prob(self, xt, ytm1, ytp1):
        pass
