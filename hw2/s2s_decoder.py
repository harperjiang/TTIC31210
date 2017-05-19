from time import time

import numpy as np

from lm_loss import LogLoss
from lstm_dataset import S2SDataSet
from lstm_graph import LSTMGraph
from ndnn.node import Dot, Embed, SoftMax, Collect, Average, MDEmbed
from ndnn.sgd import Adam
from ndnn.store import ParamStore
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.train.tsv")
dev_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.dev.tsv")
test_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.test.tsv")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

decode_graph = LSTMGraph(LogLoss(), Adam(eta=0.001, decay=0.99), dict_size, hidden_dim)

enc_lstm_graph = LSTMGraph(None, None, dict_size, hidden_dim)
enc_lstm_store = ParamStore('lstm_encoder.mdl')
enc_lstm_graph.load(enc_lstm_store.load())


def lstm_encode(data):
    enc_lstm_graph.reset()

    b_size, data_len = data.shape

    enc_lstm_graph.h0.value = np.zeros([b_size, hidden_dim])
    enc_lstm_graph.c0.value = np.zeros([b_size, hidden_dim])

    h = enc_lstm_graph.h0
    c = enc_lstm_graph.c0

    for idx in range(data_len):
        input_i = enc_lstm_graph.input()
        input_i.value = data[:, idx]
        x_i = Embed(input_i, enc_lstm_graph.embed)
        h, c = enc_lstm_graph.lstm_cell(x_i, h, c)

    enc_lstm_graph.compute()

    decode_graph.h0.value = h.value
    decode_graph.c0.value = c.value

    return decode_graph.h0, decode_graph.c0


enc_bilstm_fwd_graph = LSTMGraph(None, None, dict_size, int(hidden_dim / 2))
enc_bilstm_bcwd_graph = LSTMGraph(None, None, dict_size, int(hidden_dim / 2))
enc_bilstm_store = ParamStore('bilstm_encoder.mdl')
enc_bilstm_params = enc_bilstm_store.load()

enc_bilstm_fwd_graph.load(enc_bilstm_params[0:int(len(enc_bilstm_params) / 2)])
enc_bilstm_bcwd_graph.load(enc_bilstm_params[int(len(enc_bilstm_params) / 2):])


def bilstm_encode(data):
    enc_bilstm_fwd_graph.reset()
    enc_bilstm_bcwd_graph.reset()

    b_size, data_len = data.shape

    fwd_h = enc_bilstm_fwd_graph.h0
    fwd_c = enc_bilstm_fwd_graph.c0

    bcwd_h = enc_bilstm_bcwd_graph.h0
    bcwd_c = enc_bilstm_bcwd_graph.c0

    fwd_h.value = np.zeros([b_size, hidden_dim / 2])
    fwd_c.value = np.zeros([b_size, hidden_dim / 2])
    bcwd_h.value = np.zeros([b_size, hidden_dim / 2])
    bcwd_c.value = np.zeros([b_size, hidden_dim / 2])

    for idx in range(data_len):
        fwd_input_i = enc_bilstm_fwd_graph.input()
        fwd_input_i.value = data[:, idx]
        fwd_x_i = Embed(fwd_input_i, enc_bilstm_fwd_graph.embed)
        fwd_h, fwd_c = enc_bilstm_fwd_graph.lstm_cell(fwd_x_i, fwd_h, fwd_c)

        bcwd_input_i = enc_bilstm_bcwd_graph.input()
        bcwd_input_i.value = data[:, data_len - 1 - idx]
        bcwd_x_i = Embed(bcwd_input_i, enc_bilstm_bcwd_graph.embed)
        bcwd_h, bcwd_c = enc_bilstm_bcwd_graph.lstm_cell(bcwd_x_i, bcwd_h, bcwd_c)

    enc_bilstm_fwd_graph.compute()
    enc_bilstm_bcwd_graph.compute()

    decode_graph.h0.value = np.concatenate((fwd_h.value, bcwd_h.value), axis=1)
    decode_graph.c0.value = np.concatenate((fwd_c.value, bcwd_c.value), axis=1)
    return decode_graph.h0, decode_graph.c0


def bow_encode(data):
    h0c0 = decode_graph.input()
    h0c0.value = data

    emb = MDEmbed(h0c0, decode_graph.embed)
    avg = Average(emb)
    return avg, avg


def build_graph(batch):
    data = batch.data[1]
    decode_graph.reset()
    # Build Computation Graph according to length
    bsize, length = data.shape

    '''
    Change the function here to switch between encoders
    '''
    h, c = lstm_encode(batch.data[0])

    outputs = []
    for idx in range(length - 1):
        in_i = decode_graph.input()
        in_i.value = data[:, idx]  # Get value from batch
        x = Embed(in_i, decode_graph.embed)
        h, c = decode_graph.lstm_cell(x, h, c)
        out_i = SoftMax(Dot(h, decode_graph.v2c))
        outputs.append(out_i)
    decode_graph.output(Collect(outputs))
    decode_graph.expect(data[:, 1:])


def eval_on(dataset):
    total = 0
    accurate = 0

    for btch in dataset.batches(batch_size):
        build_graph(btch)
        bsize, length = btch.data[1].shape
        loss, predict = decode_graph.test()
        total += btch.size * (length - 1)
        accurate += predict
    return accurate / total


epoch = 100

init_dev = eval_on(dev_ds)
init_test = eval_on(test_ds)
print("Initial dev accuracy %.4f, test accuracy %.4f" % (init_dev, init_test))

best_test = 0

for i in range(epoch):

    stime = time()
    total_loss = 0
    total_acc = 0
    total_count = 0
    for batch in train_ds.batches(batch_size):
        b, l = batch.data[1].shape
        build_graph(batch)
        loss, acc = decode_graph.train()
        total_loss += loss
        total_acc += acc
        total_count += b * (l - 1)
    dev_acc = eval_on(dev_ds)
    test_acc = eval_on(test_ds)

    print("Epoch %d, "
          "time %d secs, "
          "train loss %.4f, "
          "train accuracy %.4f, "
          "dev accuracy %.4f, "
          "test accuracy %.4f" % (i, time() - stime, total_loss, total_acc / total_count, dev_acc, test_acc))

    decode_graph.update.weight_decay()
