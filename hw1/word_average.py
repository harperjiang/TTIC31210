import numpy as np
import ndnn.node as nd
import ndnn.graph as ng
import ndnn.loss as nl
import ndnn.sgd as ns
import ndnn.dataset as nds

train_file = "data/senti.binary.train"
dev_file = "data/senti.binary.dev"
test_file = "data/senti.binary.test"

# Word Vector Dimension
wv_dim = 200
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

train_ds = nds.DataSet(train_embed, train_label)
dev_ds = nds.DataSet(dev_embed, dev_label)
test_ds = nds.DataSet(test_embed, test_label)
# Build Computation Graph
graph = ng.Graph(nl.LogLoss(), ns.SGD())

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

for epoch_idx in range(epochs):
    # Train
    loss_sum = 0.0
    counter = 0
    for batch in train_ds.batches(batch_size):
        input_node.value = batch.data
        graph.expect(np.float64(batch.expect).reshape(-1, 1))
        loss, accuracy = graph.train()
        loss_sum += loss
        counter += 1
    print(loss_sum / counter)    
    # Test on Dev    
    
