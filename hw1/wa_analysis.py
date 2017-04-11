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

min_nm = np.finfo(np.float64).max()
max_nm = 0
max_w = ''
min_w = ''

for w in word_dict:
    i = word_dict[w]
    v = embed[i,:]
    nm = np.linalg(v)
    if nm > max_nm:
        max_nm = nm
        max_w = w
    if nm < min_nm:
        min_nm = nm
        min_w = w

print(max_w)
print(max_nm)

print(min_w)
print(min_nm)