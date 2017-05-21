from time import time

from ndnn.store import ParamStore
from report_stat import LogFile, ErrorStat


class Trainer(object):
    def eval_on(self, dataset):
        total = 0
        accurate = 0
        total_loss = 0

        for batch in dataset.batches(self.batch_size):
            self.graph.build_graph(batch)
            data_len = self.data_length(batch.data)
            loss, predict = self.graph.test()
            total += batch.size * (data_len - 1)
            total_loss += loss
            accurate += predict
        return total_loss / total, accurate / total

    def train(self, idx_dict, epoch, logname, graph, train_ds, dev_ds, test_ds, batch_size):
        self.graph = graph
        self.batch_size = batch_size

        logfile = LogFile(logname + ".log")
        store = ParamStore(logname + ".mdl")

        init_dev_loss, init_dev = self.eval_on(dev_ds)
        init_test_loss, init_test = self.eval_on(test_ds)
        print("Initial dev accuracy %.4f, test accuracy %.4f" % (init_dev, init_test))

        origin_time = time()

        best_acc = 0

        for i in range(epoch):
            stime = time()
            total_loss = 0
            total_pred = 0
            total = 0
            for batch in train_ds.batches(batch_size):
                graph.build_graph(batch)
                loss, predict = graph.train()
                total_loss += loss
                total_pred += predict
                data_len = self.data_length(batch.data)
                total += batch.size * (data_len - 1)

            train_time = time() - stime
            train_loss = total_loss / total
            train_acc = total_pred / total

            dev_loss, dev_acc = self.eval_on(dev_ds)
            test_loss, test_acc = self.eval_on(test_ds)

            if test_acc > best_acc:
                best_acc = test_acc
                store.store(graph.dump())

            print("Epoch %d, "
                  "time %d secs, "
                  "train loss %.4f, "
                  "dev accuracy %.4f, "
                  "test accuracy %.4f" % (i, time() - stime, total_loss, dev_acc, test_acc))

            logfile.add_record(i, time() - origin_time, train_time, train_loss, train_acc, dev_loss, dev_acc, test_acc)

            graph.update.weight_decay()

        logfile.close()

        # Collect Error Detail
        graph.loss.errorStat = ErrorStat()
        self.eval_on(dev_ds)

        print("Top 20 Error Detail:")

        for item in graph.loss.errorStat.top(20):
            print("%s,%s,%d" % (idx_dict[item[0][0]], idx_dict[item[0][1]], item[1]))

    def data_length(self, data):
        if isinstance(data, list):
            return data[1].shape[1]
        else:
            return data.shape[1]
