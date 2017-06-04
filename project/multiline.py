from common_train import Trainer
from multiline_ds import MultilineDataset
from multiline_graph import MultiLSTMEncodeGraph
from ndnn.rnn.lm_loss import LogLoss
from ndnn.sgd import Adam


hidden_dim = 200
num_line = 10

train_ds = MultilineDataset("data/ml.train", num_line)
test_ds = MultilineDataset("data/ml.test", num_line)

trainer = Trainer()

lstm_graph = MultiLSTMEncodeGraph(LogLoss(), Adam(eta=0.001, decay=0.99),
                                  len(test_ds.enc_dict), len(test_ds.dec_dict), hidden_dim, 10)
trainer.train(100, 'multiline', lstm_graph, train_ds, test_ds, test_ds, 1)
