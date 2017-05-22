import numpy as np

from lm_loss import LogLoss
from lstm_dataset import S2SDataSet
from lstm_graph import BiLSTMDecodeGraph
from ndnn.dataset import Batch
from ndnn.store import ParamStore
from vocab_dict import get_dict, translate

vocab_dict, idx_dict = get_dict()

dev_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.dev.tsv")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

graph = BiLSTMDecodeGraph(LogLoss(), dict_size, hidden_dim, 50)
store = ParamStore("model/s2s_bilstm.mdl")
graph.load(store.load())

num_sample = 10

for i in range(num_sample):
    gi = np.random.randint(0, len(dev_ds.datas))
    group = dev_ds.datas[gi]

    ii = np.random.randint(0, len(group))
    data = np.int32(group[ii][0]).reshape([1, -1])

    graph.build_graph(Batch(1, data, None))
    graph.test()

    # Collect output
    predict = graph.out.value

    print("Original sentence: " + translate(idx_dict, group[ii][0]))
    print("Generated sentence: " + translate(idx_dict, predict.flatten().tolist()))
