from common_train import Trainer
from lstm_dataset import LSTMDataSet
from lstm_graph import LogGraph
from ndnn.sgd import Adam
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.train.txt")
dev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
test_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.test.txt")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

graph = LogGraph(Adam(eta=0.001, decay=0.99), dict_size, hidden_dim)

trainer = Trainer()
trainer.train(idx_dict, 100, 'lm_logloss', graph, train_ds, dev_ds, test_ds, 50)
