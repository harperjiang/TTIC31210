from time import time

import numpy as np

from lm_loss import HingeLossOutput, HingeLoss
from lstm_dataset import LSTMDataSet
from lstm_graph import LSTMGraph
from ndnn.init import Xavier
from ndnn.node import Embed, Collect
from ndnn.sgd import Adam
from vocab_dict import get_dict
from report_stat import LogFile, SentenceLog

vocab_dict, idx_dict = get_dict()

train_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.train.txt")
dev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
test_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.test.txt")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

graph = LSTMGraph(HingeLoss(), Adam(eta=0.001), dict_size, hidden_dim)

lossEmbed = graph.param_of([dict_size, hidden_dim], Xavier())

numNegSamples = 10
negSamples = graph.input()

# negSamples.value = np.array(range(dict_size))
# v2c = graph.param_of([hidden_dim, dict_size], Xavier())

graph.resetNum = len(graph.nodes)


def build_graph(batch):
    graph.reset()
    # Build Computation Graph according to length
    bsize, length = batch.data.shape

    negSampleIdx = np.array([np.random.randint(low=0, high=dict_size) for i in range(numNegSamples)])
    negSamples.value = negSampleIdx

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
    total_loss = 0

    for batch in dataset.batches(batch_size):
        bsize, length = batch.data.shape
        build_graph(batch)
        loss, predict = graph.test()
        total += batch.size * (length - 1)
        total_loss += loss
        accurate += predict
    return total_loss / total, accurate / total


epoch = 100
enable_slog = False

logfile = LogFile('lm_hingeloss_all.log')
slog = SentenceLog('lm_hingeloss_sent_all.log')
init_dev_loss, init_dev = eval_on(dev_ds)
init_test_loss, init_test = eval_on(test_ds)
print("Initial dev accuracy %.4f, test accuracy %.4f" % (init_dev, init_test))

origin_time = time()

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
        if enable_slog:
            dev_loss, dev_acc = eval_on(dev_ds)
            slog.add_record(batch.size, dev_acc)

    train_time = time() - stime
    train_loss = total_loss / total_record
    train_acc = total_predict / total_record
    dev_loss, dev_acc = eval_on(dev_ds)
    test_loss, test_acc = eval_on(test_ds)

    print("Epoch %d, time %d secs, "
          "train time %d secs, "
          "train loss %.4f, "
          "train accuracy %.4f, "
          "dev accuracy %.4f, "
          "test accuracy %.4f" % (
              i, time() - stime, train_time, train_loss, train_acc, dev_acc,
              test_acc))
    logfile.add_record(i, time() - origin_time, train_time, train_loss, train_acc, dev_loss, dev_acc, test_acc)
    graph.update.weight_decay()

logfile.close()
slog.close()
