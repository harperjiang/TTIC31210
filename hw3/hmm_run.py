from dataset import UDDataSet
from hmm import HMM

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu')

hmm = HMM(train_ds)

adj_idx = train_ds.pos2idx('ADJ')
adj_words = sorted([(key[1], hmm.emission_counter[key]) for key in hmm.emission_counter if key[0] == adj_idx],
                   key=lambda x: x[1],
                   reverse=True)[:10]
print("\n".join(["%s & %.6f \\\\\\hline" % (train_ds.idx2word(w[0]), w[1]) for w in adj_words]))

propn_idx = train_ds.pos2idx('PROPN')
transit = sorted([(key[1], hmm.transition_counter[key]) for key in hmm.transition_counter if key[0] == propn_idx],
                 key=lambda x: x[1], reverse=True)[:5]

print("\n".join(["%s & %.6f\\\\\\hline" % (train_ds.idx2pos(t[0]), t[1]) for t in transit]))
