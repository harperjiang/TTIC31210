from dataset import UDDataSet
from hmm import HMM

train_ds = UDDataSet('data/en-ud-train.conllu')
dev_ds = UDDataSet('data/en-ud-dev.conllu')

hmm = HMM()
hmm.train(train_ds)

adj_words = sorted([(key[1], hmm.emission_counter[key]) for key in hmm.emission_counter if key[0] == 'ADJ'],
                   key=lambda x: x[1],
                   reverse=True)[:10]
print("\n".join(["%s & %.6f \\\\\\hline" % (w[0], w[1]) for w in adj_words]))

transit = sorted([(key[1], hmm.transition_counter[key]) for key in hmm.transition_counter if key[0] == 'PROPN'],
                 key=lambda x: x[1], reverse=True)[:5]

print("\n".join(["%s & %.6f\\\\\\hline" % (t[0], t[1]) for t in transit]))
