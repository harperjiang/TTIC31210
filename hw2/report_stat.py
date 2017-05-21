import operator


class LogFile:
    def __init__(self, output_file):
        self.logfile = open(output_file, 'w')
        self.logfile.write("epoch,wall_clock,train_time,train_loss,train_acc,dev_loss,dev_acc,test_acc\n")

    def add_record(self, epoch, wall_clock, train_time, train_loss, train_acc, dev_loss, dev_acc, test_acc):
        self.logfile.write(
            "%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (
                epoch, wall_clock, train_time, train_loss, train_acc, dev_loss, dev_acc, test_acc))

    def close(self):
        self.logfile.close()


class SentenceLog:
    def __init__(self, output_file):
        self.logfile = open(output_file, 'w')
        self.logfile.write('num_sentence,dev_acc\n')
        self.num_sentence = 0

    def add_record(self, n, acc):
        self.num_sentence += n
        self.logfile.write("%d,%.4f" % (self.num_sentence, acc))

    def close(self):
        self.logfile.close()


class ErrorStat:
    def __init__(self):
        self.records = {}

    def add(self, expect, actual):
        if expect != actual:
            key = (expect, actual)
            if key not in self.records:
                self.records[key] = 0
            self.records[key] += 1

    def top(self, n):
        sorted_entry = sorted(self.records.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_entry[0:n]


class SentenceDevStat:
    def __init__(self):
        pass

    def start(self):
        pass
