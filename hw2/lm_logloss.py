
import numpy as np
import ndnn
from time import time
from vocab_dict import get_dict
from ndnn.dataset import LSTMDataSet
from ndnn.graph import Graph
from ndnn.init import Xavier, Zero
from ndnn.node import Concat, Sigmoid, Add, Dot, Tanh, Mul, Embed, SoftMax, Collect
from ndnn.loss import Loss
from ndnn.sgd import Adam

vocab_dict, idx_dict = get_dict()

train_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.train.txt")
dev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
test_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.test.txt")

class LogLoss(Loss):
    def __init__(self):
        super().__init__()
    '''
    Actual is of shape [B, L, M]
    Expect is of shape [B, L]
    Should return an gradient of shape [B, L, M]    
    '''
    def loss(self, actual, expect, fortest):
        # The average loss is averaged to each slice
        all_batch_size = np.product(expect.shape)
        
        xflat = actual.reshape(-1)
        iflat = expect.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        idx = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        fetch = xflat[idx].reshape(expect.shape)
        clipval = np.maximum(fetch, ndnn.loss.clip)

        if not fortest:
            # Compute Gradient
            slgrad = -np.ones_like(expect) / (clipval * all_batch_size)
            self.grad = np.zeros_like(actual)
            self.grad.reshape(-1)[idx] = slgrad

        # Accuracy for classification is the number of corrected predicted items
        predict = np.argmax(actual, axis=-1)
        self.acc = np.equal(predict, expect).sum()

        return -np.log(clipval).mean()

graph = Graph(LogLoss(), Adam(eta=0.01, decay=0.99))

dict_size = len(vocab_dict)
hidden_dim = 300
batch_size = 50

h0 = graph.input()
c0 = graph.input()

embed = graph.param_of([dict_size, hidden_dim], Xavier())
wf = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bf = graph.param_of([hidden_dim], Zero())
wi = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bi = graph.param_of([hidden_dim], Zero())
wc = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bc = graph.param_of([hidden_dim], Zero())
wo = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bo = graph.param_of([hidden_dim], Zero())
v2c = graph.param_of([hidden_dim, dict_size], Xavier())

num_param = 12

def lstm_cell(x, h, c):
    concat = Concat(h, x)
    # Forget Gate
    f_gate = Sigmoid(Add(Dot(concat, wf), bf))
    # Input Gate
    i_gate = Sigmoid(Add(Dot(concat, wi), bi))
    # Temp Vars
    c_temp = Tanh(Add(Dot(concat, wc), bc))
    o_temp = Sigmoid(Add(Dot(concat, wo), bo))
    
    # Output
    c_next = Add(Mul(f_gate, c), Mul(i_gate, c_temp))
    h_next = Mul(o_temp, Tanh(c_next))
    return h_next, c_next
    
def build_graph(batch):
    graph.reset(num_param)
    # Build Computation Graph according to length
    bsize, length = batch.data.shape
        
    h0.value = np.zeros([bsize, hidden_dim])
    c0.value = np.zeros([bsize, hidden_dim])
        
    h = h0
    c = c0
    outputs = []
    for idx in range(length - 1):
        in_i = graph.input()
        in_i.value = batch.data[:, idx]  # Get value from batch
        x = Embed(in_i, embed)
        h, c = lstm_cell(x, h, c)
        out_i = SoftMax(Dot(h, v2c))
        outputs.append(out_i)
    graph.output(Collect(outputs))
    graph.expect(batch.data[:, 1:])
        
    
def eval_on(dataset):
    total = 0
    accurate = 0
    
    for batch in dataset.batches(batch_size):
        bsize, length = batch.data.shape
        build_graph(batch)
        loss, predict = graph.test()
        total += batch.size * (length - 1)
        accurate += predict
    return accurate / total

epoch = 100

init_dev = eval_on(dev_ds)
init_test = eval_on(test_ds)
print("Initial dev accuracy %.4f, test accuracy %.4f" % (init_dev, init_test))

for i in range(epoch):
    
    stime = time()
    
    for batch in train_ds.batches(batch_size):
        build_graph(batch)
        loss, predict = graph.train()
        
    dev_acc = eval_on(dev_ds)
    test_acc = eval_on(test_ds)


    print("Epoch %d, time %d secs, dev accuracy %.4f, test accuracy %.4f" % (i, time() - stime, dev_acc, test_acc))

    graph.update.weight_decay()
