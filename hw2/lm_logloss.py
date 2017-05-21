from time import time

import numpy as np

from lm_loss import LogLoss
from lstm_dataset import LSTMDataSet
from lstm_graph import LSTMGraph
from ndnn.node import Dot, Embed, SoftMax, Collect
from ndnn.sgd import Adam
from vocab_dict import get_dict
from report_stat import ErrorStat, LogFile, SentenceLog

vocab_dict, idx_dict = get_dict()

train_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.train.txt")
dev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
test_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.test.txt")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

graph = LSTMGraph(LogLoss(), Adam(eta=0.001, decay=0.99), dict_size, hidden_dim)


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
        out_i = SoftMax(Dot(h, graph.v2c))
        outputs.append(out_i)
    graph.output(Collect(outputs))
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

init_dev_loss, init_dev = eval_on(dev_ds)
init_test_loss, init_test = eval_on(test_ds)
print("Initial dev accuracy %.4f, test accuracy %.4f" % (init_dev, init_test))

logfile = LogFile('lm_logloss.log')
slog = SentenceLog('lm_logloss_sent.log')

origin_time = time()

for i in range(epoch):

    total_loss = 0
    total_acc = 0
    total_count = 0

    stime = time()
    for batch in train_ds.batches(batch_size):
        b, l = batch.data.shape
        build_graph(batch)
        loss, acc = graph.train()
        total_loss += loss
        total_acc += acc
        total_count += b * (l - 1)
        if enable_slog:
            dev_loss, dev_acc = eval_on(dev_ds)
            slog.add_record(batch.size, dev_acc)

    train_time = time() - stime
    train_loss = total_loss / total_count
    train_acc = total_acc / total_count

    dev_loss, dev_acc = eval_on(dev_ds)
    test_loss, test_acc = eval_on(test_ds)

    print("Epoch %d, "
          "time %d secs, "
          "train loss %.4f, "
          "train accuracy %.4f, "
          "dev accuracy %.4f, "
          "test accuracy %.4f" % (
              i, time() - stime, train_loss, train_acc, dev_acc, test_acc))

    logfile.add_record(
        i, time() - origin_time, train_time, train_loss, train_acc, dev_loss, dev_acc, test_acc)

    graph.update.weight_decay()

logfile.close()
slog.close()

# Collect Error Detail
graph.loss.errorStat = ErrorStat()
eval_on(dev_ds)

print("Top 20 Error Detail:")

for item in graph.loss.errorStat.top(20):
    print("%s,%s,%d" % (idx_dict[item[0][0]], idx_dict[item[0][1]], item[1]))
