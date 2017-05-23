class UDSentence(object):
    def __init__(self):
        self._words = []

    def add_word(self, word, pos):
        self._words.append((word, pos))

    def __len__(self):
        return len(self._words)

    def words(self):
        return self._words


class UDDataSet(object):
    def __init__(self, filename):

        self._sentences = []
        self.pos = set()
        self.words = set()

        sentence = UDSentence()
        for line in open(filename, 'r').readlines():
            if not line.startswith("#"):
                if len(line.strip()) == 0:  # Separator
                    if len(sentence) != 0:
                        self._sentences.append(sentence)
                    sentence = UDSentence()
                else:
                    words = line.split()
                    sentence.add_word(words[1], words[3])
                    self.pos.add(words[3])
                    self.words.add(words[1])

    def sentences(self):
        return self._sentences
