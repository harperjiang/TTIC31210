import numpy as np

from lm_loss import LogLoss
from lstm_dataset import LSTMDataSet, S2SDataSet
from lstm_graph import LSTMEncodeGraph, BiLSTMEncodeGraph, BowEncodeGraph
from ndnn.dataset import Batch
from ndnn.sgd import Adam
from ndnn.store import ParamStore
from vocab_dict import get_dict, translate

vocab_dict, idx_dict = get_dict()

lmdev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
s2strain_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.train.tsv")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

lstm_encode_graph = LSTMEncodeGraph(LogLoss(), Adam(eta=0.001), dict_size, hidden_dim)
lstm_encode_store = ParamStore("model/s2s_lstm.mdl")
lstm_encode_graph.load(lstm_encode_store.load())

bilstm_encode_graph = BiLSTMEncodeGraph(LogLoss(), Adam(eta=0.001), dict_size, hidden_dim)
bilstm_encode_store = ParamStore("model/s2s_bilstm.mdl")
bilstm_encode_graph.load(bilstm_encode_store.load())

bow_encode_graph = BowEncodeGraph(LogLoss(), Adam(eta=0.001), dict_size, hidden_dim)
bow_encode_store = ParamStore("model/s2s_bow.mdl")
bow_encode_graph.load(bow_encode_store.load())

encode_graphs = [lstm_encode_graph, bilstm_encode_graph, bow_encode_graph]

train_sentences = []
train_encoded = []

# Encode Record from S2STrain
first_round = True
for batch in s2strain_ds.batches(batch_size):
    for gi, encode_graph in enumerate(encode_graphs):
        encode_graph.reset()
        encode_graph.build_graph(batch)
        encode_graph.test()
        encoded = encode_graph.encode_result()

        if first_round:
            train_encoded.append(encoded)
        else:
            train_encoded[gi] = np.concatenate((train_encoded[gi], encoded), axis=0)
    train_sentences = train_sentences + batch.data[0].tolist()
    first_round = False
input_size = 10

dev_sentences = []
dev_encoded = []

# Encode Random Record from LMDev
first_round = True

for i in range(input_size):
    dg_idx = np.random.randint(0, len(lmdev_ds.datas))
    data_group = lmdev_ds.datas[dg_idx]
    data_idx = np.random.randint(0, len(data_group))
    dev_sentences.append(np.int32(data_group[data_idx]))
    data = np.int32(data_group[data_idx]).reshape([1, -1])

    for gi, encode_graph in enumerate(encode_graphs):
        encode_graph.reset()
        encode_graph.build_graph(Batch(1, [data, data], None))
        encode_graph.test()
        encoded = encode_graph.encode_result()

        if first_round:
            dev_encoded.append(encoded)
        else:
            dev_encoded[gi] = np.concatenate((dev_encoded[gi], encoded), axis=0)
    first_round = False

num_nearest_neighbor = 10
# Compute Cosine Similarity
for gi in range(len(encode_graphs)):
    dot = np.einsum('tn,dn->td', train_encoded[gi], dev_encoded[gi])
    train_norm = np.linalg.norm(train_encoded[gi], axis=1)
    dev_norm = np.linalg.norm(dev_encoded[gi], axis=1)

    sim = ((dot / dev_norm).T / train_norm)

    # Print Nearest Neighbors
    for i in range(input_size):
        nearest_neighbor = np.argsort(-sim[i, :])[0:num_nearest_neighbor]
        print("Graph %d" % (gi))
        print("Original Sentence:" + translate(idx_dict, dev_sentences[i]))
        print("Nearest Neighbors:")
        for j in range(num_nearest_neighbor):
            print(translate(idx_dict, train_sentences[nearest_neighbor[j]]))
