from time import time

import numpy as np

from logloss import LogLoss
from lstm_graph import LSTMGraph
from lstm_dataset import S2SDataSet
from ndnn.node import Concat, Sigmoid, Add, Dot, Tanh, Mul, Embed, SoftMax, Collect
from ndnn.sgd import Adam
from ndnn.store import ParamStore
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.train.tsv")
dev_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.dev.tsv")
test_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.test.tsv")

dict_size = len(vocab_dict)
hidden_dim = 300
batch_size = 50

graph = LSTMGraph(LogLoss(), Adam(eta=0.01, decay=0.99), dict_size, hidden_dim)

num_param = 12

def build_graph(batch):
    data = batch.data[0]
    graph.reset(num_param)
    # Build Computation Graph according to length
    bsize, length = data.shape

    graph.h0.value = np.zeros([bsize, hidden_dim])
    graph.c0.value = np.zeros([bsize, hidden_dim])

    h = graph.h0
    c = graph.c0
    outputs = []
    for idx in range(length - 1):
        in_i = graph.input()
        in_i.value = data[:, idx]  # Get value from batch
        x = Embed(in_i, graph.embed)
        h, c = graph.lstm_cell(x, h, c)
        out_i = SoftMax(Dot(h, graph.v2c))
        outputs.append(out_i)
    graph.output(Collect(outputs))
    graph.expect(data[:, 1:])


def eval_on(dataset):
    total = 0
    accurate = 0

    for batch in dataset.batches(batch_size):
        build_graph(batch)
        bsize, length = batch.data[0].shape
        loss, predict = graph.test()
        total += batch.size * (length - 1)
        accurate += predict
    return accurate / total


epoch = 100

init_dev = eval_on(dev_ds)
init_test = eval_on(test_ds)
print("Initial dev accuracy %.4f, test accuracy %.4f" % (init_dev, init_test))

best_test = 0
store = ParamStore('lstm_encoder.mdl')

for i in range(epoch):

    stime = time()
    total_loss = 0
    for batch in train_ds.batches(batch_size):
        build_graph(batch)
        loss, predict = graph.train()
        total_loss += loss
    dev_acc = eval_on(dev_ds)
    test_acc = eval_on(test_ds)

    if test_acc > best_test:
        best_test = test_acc
        store.store(graph.params)

    print("Epoch %d, "
          "time %d secs, "
          "train loss %.4f, "
          "dev accuracy %.4f, "
          "test accuracy %.4f" % (i, time() - stime, total_loss, dev_acc, test_acc))

    graph.update.weight_decay()
