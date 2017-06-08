from common_train import Trainer
from ndnn.rnn.lm_loss import LogLoss
from ndnn.rnn.lstm_dataset import S2SDict, S2SDataSet
from ndnn.rnn.lstm_graph import BiLSTMDecodeGraph
from ndnn.store import ParamStore

dict = S2SDict(["data/part.train", "data/whole.test"])

test_ds = S2SDataSet(dict.enc_dict, dict.dec_dict, "data/whole.eval")

hidden_dim = 100
batch_size = 50

trainer = Trainer()

lstm_graph = BiLSTMDecodeGraph(LogLoss(), len(dict.enc_dict), len(dict.dec_dict), hidden_dim, 10)
store = ParamStore("part_part.mdl")
lstm_graph.load(store.load())

counter = 0

for batch in test_ds.batches(1):
    if counter > 10:
        break
    counter += 1
    lstm_graph.build_graph(batch)
    lstm_graph.test()
    print(dict.enc_translate(batch.data[0]))
    print(dict.dec_translate(lstm_graph.out.value))
