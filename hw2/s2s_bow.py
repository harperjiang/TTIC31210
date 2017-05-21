from common_train import Trainer
from lm_loss import LogLoss
from lstm_dataset import S2SDataSet
from lstm_graph import BowEncodeGraph
from ndnn.sgd import Adam
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.train.tsv")
dev_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.dev.tsv")
test_ds = S2SDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.seq2seq.test.tsv")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

trainer = Trainer()

graph = BowEncodeGraph(LogLoss(), Adam(eta=0.001, decay=0.99), dict_size, hidden_dim)
trainer.train(idx_dict, 100, 's2s_bow', graph, train_ds, dev_ds, test_ds, 50)
