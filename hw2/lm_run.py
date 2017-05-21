from common_train import Trainer
from lstm_dataset import LSTMDataSet
from lstm_graph import HingeGraph, LogGraph
from ndnn.sgd import Adam
from vocab_dict import get_dict

vocab_dict, idx_dict = get_dict()

train_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.train.txt")
dev_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.dev.txt")
test_ds = LSTMDataSet(vocab_dict, idx_dict, "bobsue-data/bobsue.lm.test.txt")

dict_size = len(vocab_dict)
hidden_dim = 200
batch_size = 50

trainer = Trainer()

loss_graph = LogGraph(Adam(eta=0.001), dict_size, hidden_dim)
trainer.train(idx_dict, 100, 'lm_logloss', loss_graph, train_ds, dev_ds, test_ds, 50)

# Share Embedding
sem_graph = HingeGraph(Adam(eta=0.001), dict_size, hidden_dim, -1, False)
trainer.train(idx_dict, 100, 'lm_hingeloss_sem', sem_graph, train_ds, dev_ds, test_ds, 50)

all_graph = HingeGraph(Adam(eta=0.001), dict_size, hidden_dim, -1, True)
trainer.train(idx_dict, 100, 'lm_hingeloss_all', all_graph, train_ds, dev_ds, test_ds, 50)

r100_graph = HingeGraph(Adam(eta=0.001), dict_size, hidden_dim, 100, True)
trainer.train(idx_dict, 100, 'lm_hingeloss_r100', all_graph, train_ds, dev_ds, test_ds, 50)

r10_graph = HingeGraph(Adam(eta=0.001), dict_size, hidden_dim, 10, True)
trainer.train(idx_dict, 100, 'lm_hingeloss_r10', all_graph, train_ds, dev_ds, test_ds, 50)
