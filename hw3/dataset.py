class UDSentence(object):
    def __init__(self):
        self.words = []

    def add_word(self, word, pos):
        self.words.append((word, pos))

    def __len__(self):
        return len(self.words)


UNK = -1


class UDDataSet(object):
    def __init__(self, filename, ref=None):

        self._sentences = []
        self.pos = {}
        self.words = {}
        self.idxwords = []
        self.idxpos = []

        if ref is not None:
            self.pos = ref.pos
            self.words = ref.words
            self.idxwords = ref.idxwords
            self.idxpos = ref.idxpos

        sentence = UDSentence()
        file = open(filename, 'r')
        for line in file.readlines():
            if not line.startswith("#"):
                if len(line.strip()) == 0:  # Separator
                    if len(sentence) != 0:
                        self._sentences.append(sentence)
                    sentence = UDSentence()
                else:
                    words = line.split()
                    word_idx = self.word2idx(words[1], ref is None)
                    pos_idx = self.pos2idx(words[3], ref is None)
                    sentence.add_word(word_idx, pos_idx)
        file.close()

    def sentences(self):
        return self._sentences

    def word2idx(self, word, add_unknown):
        if add_unknown and word not in self.words:
            self.words[word] = len(self.words)
            self.idxwords.append(word)
        return self.words.get(word, UNK)

    def pos2idx(self, pos, add_unknown):
        if add_unknown and pos not in self.pos:
            self.pos[pos] = len(self.pos)
            self.idxpos.append(pos)
        # Should throw error for unknown pos
        return self.pos[pos]

    def idx2word(self, idx):
        return self.idxwords[idx]

    def idx2pos(self, idx):
        return self.idxpos[idx]

    def lookup_word(self, word):
        return self.words.get(word, UNK)

    def lookup_pos(self, pos):
        return self.pos[pos]
