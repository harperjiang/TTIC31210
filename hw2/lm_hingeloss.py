import numpy as np
from time import time

from ndnn.dataset import LSTMDataSet
from ndnn.graph import Graph
from ndnn.init import Xavier, Zero
from ndnn.loss import TrivialLoss
from ndnn.node import Node, Concat, Sigmoid, Add, Dot, Tanh, Mul, Embed, Collect
from ndnn.sgd import Adam
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.train.txt")
dev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
test_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.test.txt")


class HingeLoss(Node):
    def __init__(self, actual, expect, negSamples):
        super().__init__([actual])
        self.actual = actual
        self.expect = expect
        self.negSamples = negSamples

    def compute(self):
        ytht = np.einsum('ij,ij->i', self.actual.value, self.expect.value)
        ypht = np.matmul(self.actual.value, self.negSamples.value.T)
        value = np.maximum(1 - ytht[:, np.newaxis] + ypht, 0)
        self.mask = value > 0
        return value.sum(axis=1)

    def updateGrad(self):
        gradscalar = self.grad[:, np.newaxis]
        multiplier = self.mask.sum(axis=1, keepdims=True)
        self.actual.grad += gradscalar * (
            np.matmul(self.mask, self.negSamples.value) - multiplier * self.expect.value)
        self.expect.grad += gradscalar * multiplier * (-self.actual.value)
        self.negSamples.grad += np.einsum('br,bh->rh', self.mask, gradscalar * self.actual.value)

class HingePredict(Node):
    def __init__(self, actual, allEmbed, negSamples):
        super().__init__([actual])
        self.actual = actual
        self.allEmbed = allEmbed
        self.negSamples = negSamples

    def compute(self):
        # Find the one with smallest loss as prediction

        # All_embed has size D,H
        # Actual has size B,H
        # Neg Samples has size R,H

        d = self.allEmbed.value.shape[0]
        r = self.negSamples.value.shape[0]

        bd = np.einsum('bh,dh->bd', self.actual.value, self.allEmbed.value)
        br = np.einsum('bh,rh->br', self.actual.value, self.negSamples.value)

        bdr1 = np.repeat(bd[:, :, np.newaxis], r, axis=2)
        bdr2 = np.repeat(br[:, np.newaxis, :], d, axis=1)

        predict = np.maximum(1 - bdr1 + bdr2, 0).sum(axis=2).argmin(axis=1)
        return predict


    def updateGrad(self):
        raise Exception("Operation not supported")


graph = Graph(TrivialLoss(), Adam(eta=0.01, decay=0.99))

dict_size = len(vocab_dict)
hidden_dim = 300
batch_size = 50

h0 = graph.input()
c0 = graph.input()

embed = graph.param_of([dict_size, hidden_dim], Xavier())
#lossEmbed = graph.param_of([dict_size, hidden_dim], Xavier())

wf = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bf = graph.param_of([hidden_dim], Zero())
wi = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bi = graph.param_of([hidden_dim], Zero())
wc = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bc = graph.param_of([hidden_dim], Zero())
wo = graph.param_of([2 * hidden_dim, hidden_dim], Xavier())
bo = graph.param_of([hidden_dim], Zero())

numNegSamples = 100
negSampleIdx = np.array([np.random.randint(low=0, high=dict_size) for i in range(numNegSamples)])

negSamples = graph.input()
negSamples.value = negSampleIdx
# v2c = graph.param_of([hidden_dim, dict_size], Xavier())

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


def build_train_graph(batch):
    graph.reset(num_param)
    # Build Computation Graph according to length
    bsize, length = batch.data.shape

    neg = Embed(negSamples, embed)

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

        expect_i = graph.input()
        expect_i.value = batch.data[:, idx + 1]
        embed_expect = Embed(expect_i, embed)

        loss = HingeLoss(h, embed_expect, neg)
        out_i = loss
        outputs.append(out_i)

    graph.output(Collect(outputs))
    graph.expect(batch.data[:, 1:])

def build_predict_graph(batch):
    graph.reset(num_param)
    # Build Computation Graph according to length
    bsize, length = batch.data.shape

    neg = Embed(negSamples, embed)

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

        predict_i = HingePredict(h, embed, neg)
        outputs.append(predict_i)

    graph.output(Collect(outputs))
    graph.expect(batch.data[:, 1:])

def eval_on(dataset):
    total = 0
    accurate = 0

    for batch in dataset.batches(batch_size):
        bsize, length = batch.data.shape
        build_predict_graph(batch)
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
    total_loss = 0
    for batch in train_ds.batches(batch_size):
        build_train_graph(batch)
        loss, predict = graph.train()
        total_loss += loss

    dev_acc = eval_on(dev_ds)
    test_acc = eval_on(test_ds)

    print("Epoch %d, time %d secs, "
          "train loss %.4f, "
          "dev accuracy %.4f, "
          "test accuracy %.4f" % (i, time() - stime, total_loss, dev_acc, test_acc))

    graph.update.weight_decay()
