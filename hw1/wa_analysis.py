import ndnn.store as ns
import numpy as np


train_file = "data/senti.binary.train"
dev_file = "data/senti.binary.dev"
test_file = "data/senti.binary.test"

# Word Vector Dimension
word_dict = {}

lines = open(train_file, "rb").readlines()

for line in lines:
    pieces = line.split()
    label = int(pieces[-1])
    words = pieces[0:-1]
    for word in words:
        if word not in word_dict:
            word_dict[word] = len(word_dict)

store = ns.ParamStore('word_avg.mdl')

embed = store.load()[0]

d = embed.shape[0]

norms = [0.0] * d

for w in word_dict:
    i = word_dict[w]
    v = embed[i, :]
    norms[i] = (np.linalg.norm(v), w)
    
norms.sort(key=lambda x: x[0])

print(norms[:10])
print(norms[:-10])