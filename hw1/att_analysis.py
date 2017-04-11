import numpy as np
import ndnn.node as nd
import ndnn.graph as ng
import ndnn.loss as nl
import ndnn.sgd as ns
import ndnn.dataset as nds
import ndnn.store as nst

train_file = "data/senti.binary.train"
dev_file = "data/senti.binary.dev"
test_file = "data/senti.binary.test"

model_file = "attention.mdl"
store = nst.ParamStore(model_file)
params = store.load()

embed = params[0]
att_w = params[1]

# Word Vector Dimension
word_dict = {}
word_variance = {}
def load(file_name):
    lines = open(file_name, "rb").readlines()
    for line in lines:
        pieces = line.split()
        words = pieces[0:-1]
        for word in words:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
                

# Load data
load(train_file)
load(dev_file)
load(test_file)


lines = open(train_file, "rb").readlines()
for line in lines:
    pieces = line.split()
    words = pieces[0:-1]
    for word in words:
        if word not in word_variance:
            word_variance[word] = []
    word_idx = [word_dict[word] for w in words]
    word_embed = embed[word_idx, :]
    word_map = np.matmul(word_embed, att_w)
    lmax = np.max(word_map, axis=-1, keepdims=True)
    ex = np.exp(word_map - lmax)
    softmax = ex / np.sum(ex, axis=-1, keepdims=True) 

    for idx, word in enumerate(words):
        word_variance[word].append(softmax[idx])
        
variances = []        
        
for w in word_variance:
    data = word_variance[w]
    var = np.var(data)
    variances.append((var, w))
    
variances.sort(key=lambda x:x[0])

print(variances[:5])
print(variances[-5:])



