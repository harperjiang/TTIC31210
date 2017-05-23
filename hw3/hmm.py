from dataset import UDDataSet

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu')

# Train HMM by doing counting

emission_counter = {}
transition_counter = {}

for sent in train_ds.sentences():
    prev_word = None
    for word in sent.words():
        emit_key = (word[1], word[0])
        if emit_key not in emission_counter:
            emission_counter[emit_key] = 0
        emission_counter[emit_key] += 1
        if prev_word is not None:
            transit_key = (prev_word[1], word[1])
            if transit_key not in transition_counter:
                transition_counter[transit_key] = 0
            transition_counter[transit_key] += 1
        prev_word = word

# Smoothing
for pos in train_ds.pos:
    emission_counter[(pos, "<UNK>")] = 1


# Normalize
def normalize(counter):
    sum_val = sum(v for v in counter.values())
    for key in counter:
        counter[key] /= sum_val


normalize(emission_counter)
normalize(transition_counter)

adj_words = sorted([(key[1], emission_counter[key]) for key in emission_counter if key[0] == 'ADJ'], key=lambda x: x[1],
                   reverse=True)[:10]
print("\n".join(["%s & %.6f \\\\\\hline" % (w[0], w[1]) for w in adj_words]))

transit = sorted([(key[1], transition_counter[key]) for key in transition_counter if key[0] == 'PROPN'],
                 key=lambda x: x[1], reverse=True)[:5]

print("\n".join(["%s & %.6f\\\\\\hline" % (t[0], t[1]) for t in transit]))
