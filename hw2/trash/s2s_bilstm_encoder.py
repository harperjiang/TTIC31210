from time import time

import numpy as np

from lm_loss import LogLoss
from lstm_dataset import S2SDataSet
from lstm_graph import LSTMGraph
from ndnn.node import Dot, Embed, SoftMax, Collect
from ndnn.sgd import Adam
from ndnn.store import ParamStore
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.train.tsv")
dev_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.dev.tsv")
test_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.test.tsv")

dict_size = len(vocab_dict)
hidden_dim = 100
batch_size = 50

fwd_graph = LSTMGraph(LogLoss(), Adam(eta=0.001, decay=0.99), dict_size, hidden_dim)
bcwd_graph = LSTMGraph(LogLoss(), Adam(eta=0.001, decay=0.99), dict_size, hidden_dim)


def build_graph(batch):
    data = batch.data[0]

    fwd_graph.reset()
    bcwd_graph.reset()

    bsize, length = data.shape

    fwd_graph.h0.value = np.zeros([bsize, hidden_dim])
    fwd_graph.c0.value = np.zeros([bsize, hidden_dim])
    bcwd_graph.h0.value = np.zeros([bsize, hidden_dim])
    bcwd_graph.c0.value = np.zeros([bsize, hidden_dim])

    fwd_h = fwd_graph.h0
    fwd_c = fwd_graph.c0
    bcwd_h = bcwd_graph.h0
    bcwd_c = bcwd_graph.c0

    fwd_outputs = []
    bcwd_outputs = []
    for idx in range(length - 1):
        # Build Fowward Graph
        fwd_in_i = fwd_graph.input()
        fwd_in_i.value = data[:, idx]  # Get value from batch
        fwd_x = Embed(fwd_in_i, fwd_graph.embed)
        fwd_h, fwd_c = fwd_graph.lstm_cell(fwd_x, fwd_h, fwd_c)
        fwd_out_i = SoftMax(Dot(fwd_h, fwd_graph.v2c))
        fwd_outputs.append(fwd_out_i)

        # Build Backward Graph
        bcwd_in_i = bcwd_graph.input()
        bcwd_in_i.value = data[:, length - 1 - idx]  # Get value from batch
        bcwd_x = Embed(bcwd_in_i, bcwd_graph.embed)
        bcwd_h, bcwd_c = bcwd_graph.lstm_cell(bcwd_x, bcwd_h, bcwd_c)
        bcwd_out_i = SoftMax(Dot(bcwd_h, bcwd_graph.v2c))
        bcwd_outputs.append(bcwd_out_i)

    fwd_graph.output(Collect(fwd_outputs))
    fwd_graph.expect(data[:, 1:])

    bcwd_graph.output(Collect(bcwd_outputs))
    bcwd_graph.expect(np.flip(data, axis=1)[:, 1:])


def eval_on(dataset):
    total = 0
    accurate = 0

    for batch in dataset.batches(batch_size):
        build_graph(batch)
        bsize, length = batch.data[0].shape
        fwd_loss, fwd_predict = fwd_graph.test()
        total += batch.size * (length - 1)
        accurate += fwd_predict
        bcwd_loss, bcwd_predict = bcwd_graph.test()
        total += batch.size * (length - 1)
        accurate += bcwd_predict

    return accurate / total


epoch = 100

init_dev = eval_on(dev_ds)
init_test = eval_on(test_ds)
print("Initial dev accuracy %.4f, test accuracy %.4f" % (init_dev, init_test))

best_test = 0
store = ParamStore('bilstm_encoder.mdl')

for i in range(epoch):

    stime = time()
    total_loss = 0
    for batch in train_ds.batches(batch_size):
        build_graph(batch)
        fwd_loss, predict = fwd_graph.train()
        total_loss += fwd_loss
        bcwd_loss, predict = bcwd_graph.train()
        total_loss += bcwd_loss
    dev_acc = eval_on(dev_ds)
    test_acc = eval_on(test_ds)

    if test_acc > best_test:
        best_test = test_acc
        store.store(fwd_graph.dump()+ bcwd_graph.dump())

    print("Epoch %d, "
          "time %d secs, "
          "train loss %.4f, "
          "dev accuracy %.4f, "
          "test accuracy %.4f" % (i, time() - stime, total_loss, dev_acc, test_acc))

    fwd_graph.update.weight_decay()
    bcwd_graph.update.weight_decay()
