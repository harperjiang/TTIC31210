from time import time

import numpy as np

from lm_loss import HingeLossOutput, HingeLoss
from lstm_dataset import LSTMDataSet
from lstm_graph import LSTMGraph
from ndnn.init import Xavier
from ndnn.node import Embed, Collect
from ndnn.sgd import Adam
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.train.txt")
dev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
test_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.test.txt")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

graph = LSTMGraph(HingeLoss(), Adam(eta=0.001), dict_size, hidden_dim)

lossEmbed = graph.param_of([dict_size, hidden_dim], Xavier())

numNegSamples = 200
negSampleIdx = np.array([np.random.randint(low=0, high=dict_size) for i in range(numNegSamples)])

negSamples = graph.input()
# negSamples.value = negSampleIdx
negSamples.value = np.array(range(dict_size))
# v2c = graph.param_of([hidden_dim, dict_size], Xavier())

graph.resetNum = len(graph.nodes)


def build_graph(batch):
    graph.reset()
    # Build Computation Graph according to length
    bsize, length = batch.data.shape

    graph.h0.value = np.zeros([bsize, hidden_dim])
    graph.c0.value = np.zeros([bsize, hidden_dim])

    h = graph.h0
    c = graph.c0
    outputs = []
    for idx in range(length - 1):
        in_i = graph.input()
        in_i.value = batch.data[:, idx]  # Get value from batch
        x = Embed(in_i, graph.embed)
        h, c = graph.lstm_cell(x, h, c)

        outputs.append(h)

    graph.output(HingeLossOutput(Collect(outputs), lossEmbed, negSamples))
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
    total_loss = 0
    total_predict = 0
    total_record = 0
    for batch in train_ds.batches(batch_size):
        build_graph(batch)
        loss, predict = graph.train()
        total_loss += loss
        total_predict += predict
        total_record += batch.size * (batch.data.shape[1] - 1)
    dev_acc = eval_on(dev_ds)
    test_acc = eval_on(test_ds)

    print("Epoch %d, time %d secs, "
          "train loss %.4f, "
          "train accuracy %.4f, "
          "dev accuracy %.4f, "
          "test accuracy %.4f" % (
          i, time() - stime, total_loss / total_record, total_predict / total_record, dev_acc, test_acc))

    graph.update.weight_decay()
