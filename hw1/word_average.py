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

model_file = "word_avg.mdl"
store = nst.ParamStore(model_file)
# Word Vector Dimension
wv_dim = 100
word_dict = {}

def load(file_name):
    lines = open(file_name, "rb").readlines()
    emds = []
    labels = []
    for line in lines:
        pieces = line.split()
        label = int(pieces[-1])
        words = pieces[0:-1]
        for word in words:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
        emd = np.float64([word_dict[w] for w in words])
        emds.append(emd)
        labels.append(label)
    
    return emds, labels

# Load data
train_embed, train_label = load(train_file)
dev_embed, dev_label = load(dev_file)
test_embed, test_label = load(test_file)

train_ds = nds.VarLenDataSet(train_embed, train_label)
dev_ds = nds.VarLenDataSet(dev_embed, dev_label)
test_ds = nds.VarLenDataSet(test_embed, test_label)
# Build Computation Graph
graph = ng.Graph(nl.LogLoss(), ns.Adam(eta=0.01))

# Word Embedding Matrix using Xavier
word_embedding = graph.param_of([len(word_dict), wv_dim])
# Weight vector
weight = graph.param_of([wv_dim, 1])

input_node = graph.input()
embed = nd.Embed(input_node, word_embedding)
average = nd.Average(embed)
dot = nd.Dot(average, weight)
sigmoid = nd.Sigmoid(dot)
graph.output(sigmoid)

epochs = 100
batch_size = 50

def train():
    loss_sum = 0.0
    batch_counter = 0
    for batch in train_ds.batches(batch_size):
        input_node.value = batch.data
        graph.expect(np.float64(batch.expect).reshape(-1, 1))
        loss, accuracy = graph.train()
        loss_sum += loss
        batch_counter += 1
    return loss_sum / batch_counter

def evaluate():
    loss_sum = 0.0
    batch_counter = 0
    for batch in dev_ds.batches(batch_size):
        input_node.value = batch.data
        graph.expect(np.float64(batch.expect).reshape(-1, 1))
        loss, accuracy = graph.test()
        loss_sum += loss
        batch_counter += 1
    return loss_sum / batch_counter
    
def test():
    acc_sum = 0
    item_counter = 0
    for batch in test_ds.batches(batch_size):
        input_node.value = batch.data
        graph.expect(np.float64(batch.expect).reshape(-1, 1))
        loss, accuracy = graph.test()
        acc_sum += accuracy
        item_counter += batch.size
    return acc_sum / item_counter    
    
best_dev = np.finfo(np.float64).max

# Load parameters if exists
params = store.load()
if params is not None:
    word_embedding.value = params[0]
    weight.value = params[1]

dev_loss = evaluate()
print("Initial Dev Loss is {0:f}".format(dev_loss))
test_accuracy = test()
print("Initial Test accuracy is {0:f}".format(test_accuracy))

for epoch_idx in range(epochs):
    # Train
    train_loss = train()
    print("Epoch {0:d}, training Loss is {1:f}".format(epoch_idx, train_loss))    
    # Eval on Dev    
    dev_loss = evaluate()
    print("Dev loss is {0:f}".format(dev_loss))
    
    test_accuracy = test()
    print("Test accuracy is {0:f}".format(test_accuracy))
    
    
    if(dev_loss < best_dev):
        best_dev = dev_loss
        # Save Model
        store.store([word_embedding.value, weight.value])
    else:
        # Early stop?
        pass

test_accuracy = test()
print("Test accuracy is {0:f}".format(test_accuracy))
