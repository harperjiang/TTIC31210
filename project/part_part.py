from common_train import Trainer
from ndnn.rnn.lm_loss import LogLoss
from ndnn.rnn.lstm_dataset import S2SDict, S2SDataSet
from ndnn.rnn.lstm_graph import BiLSTMEncodeGraph
from ndnn.sgd import Adam

dict = S2SDict(["data/part.train"])

train_ds = S2SDataSet(dict.enc_dict, dict.dec_dict, "data/part.train")
test_ds = S2SDataSet(dict.enc_dict, dict.dec_dict, "data/part.test")

hidden_dim = 200
batch_size = 50

trainer = Trainer()

lstm_graph = BiLSTMEncodeGraph(LogLoss(), Adam(eta=0.001, decay=0.99),
                               len(dict.enc_dict), len(dict.dec_dict), hidden_dim)
trainer.train(100, 'part_part', lstm_graph, train_ds, test_ds, test_ds, 50)
