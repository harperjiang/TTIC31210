BOS = "BOS"
EOS = "EOS"
UNK = "UNK"


class UDSentence(object):
    def __init__(self):
        self._words = []

    def add_word(self, word, pos):
        self._words.append((word, pos))

    def words(self):
        return self._words

    def __len__(self):
        return len(self._words)


class UDDataSet(object):
    def __init__(self, filename):

        self._sentences = []
        self.pos = {}
        self.words = {}
        self.idxwords = []
        self.idxpos = []

        self.pos2idx(BOS)
        self.pos2idx(EOS)
        self.word2idx(UNK)

        sentence = UDSentence()
        for line in open(filename, 'r').readlines():
            if not line.startswith("#"):
                if len(line.strip()) == 0:  # Separator
                    if len(sentence) != 0:
                        self._sentences.append(sentence)
                    sentence = UDSentence()
                else:
                    words = line.split()
                    word_idx = self.word2idx(words[1])
                    pos_idx = self.pos2idx(words[3])
                    sentence.add_word(word_idx, pos_idx)

    def sentences(self):
        return self._sentences

    def word2idx(self, word):
        if word not in self.words:
            self.words[word] = len(self.words)
            self.idxwords.append[word]
        return self.words[word]

    def pos2idx(self, pos):
        if pos not in self.pos:
            self.pos[pos] = len(self.pos)
            self.idxpos.append(pos)
        return self.pos[pos]

    def lookup_word(self, word):
        if word in self.words:
            return self.words[word]
        return -1
