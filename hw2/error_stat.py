import operator

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
