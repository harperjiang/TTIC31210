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

model_file = "attention_var2.mdl"
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
graph = ng.Graph(nl.LogLoss(), ns.SGD(eta=0.05))

# Word Embedding Matrix using Xavier
word_embedding = graph.param_of([len(word_dict), wv_dim])
# Weight vector
weight = graph.param_of([wv_dim, 1])
attention_weight = graph.param_of([wv_dim])


class EmbedMap(nd.Node):
    def __init__(self, embed, weight):
        super(EmbedMap, self).__init__([embed, weight])
        self.embed = embed
        self.weight = weight
        self.window = [.1, .2, .4, .2, .1]

    def compute(self):
        raw = np.einsum("bld,d->bl", self.embed.value, self.weight.value)
        res = np.zeros_like(raw)
        b = raw.shape[0]
        l = raw.shape[1]

        # Pad the result and do convolution
        raw_padded = np.zeros([b, l + 4])
        raw_padded[:, 2:-2] = raw

        for i in range(l):
            res[:, i] += np.matmul(raw_padded[:, i:i + 5], self.window)
        return res

    def updateGrad(self):
        # convolution
        b = self.grad.shape[0]
        l = self.grad.shape[1]
        padded_grad = np.zeros([b, l + 4])
        padded_grad[:, 2:-2] = self.grad
        grad = np.zeros_like(self.grad)

        for i in range(l):
            grad[:, i] += np.matmul(padded_grad[:, i:i + 5], self.window)

        self.embed.grad += np.einsum("bl,d->bld", grad, self.weight.value)
        self.weight.grad += np.einsum("bl,bld->d", grad, self.embed.value)


class Attention(nd.Node):
    def __init__(self, embed, smax):
        super(Attention, self).__init__([embed, smax])
        self.embed = embed
        self.smax = smax

    def compute(self):
        return np.einsum("bld,bl->bd", self.embed.value, self.smax.value)

    def updateGrad(self):
        self.embed.grad += np.einsum("bd,bl->bld", self.grad, self.smax.value)
        self.smax.grad += np.einsum("bd,bld->bl", self.grad, self.embed.value)


input_node = graph.input()
embed = nd.Embed(input_node, word_embedding)
mapped = EmbedMap(embed, attention_weight)
softmax = nd.SoftMax(mapped)
attention = Attention(embed, softmax)
dot = nd.Dot(attention, weight)
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
    acc_sum = 0.0
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
    attention_weight.value = params[1]
    weight.value = params[2]

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

    # Save Model
    if (dev_loss < best_dev):
        best_dev = dev_loss
        store.store([word_embedding.value, attention_weight.value, weight.value])
    else:
        # Early stop
        pass

test_accuracy = test()
print("Test accuracy is {0:f}".format(test_accuracy))
