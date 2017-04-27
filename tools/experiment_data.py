from collections import namedtuple

Word = namedtuple('Word', ['word', 'pos', 'head_idx', 'dep_label'])
Subword = namedtuple('Subword', ['subword', 'word_idx'])


class ExperimentData:
    def __init__(self, data_path, subword_data_path):
        self.data = ExperimentData.read_data(data_path)
        self.subword = self.read_subword_data(subword_data_path)

    @staticmethod
    def read_data(data_path):
        data = list()
        sentence_data = list()

        with open(data_path) as data_file:
            for line in data_file:
                if len(line) <= 1:
                    data.append(sentence_data.copy())
                    sentence_data.clear()
                elif line[0] != '#':
                    token = line.strip().split()
                    sentence_data.append(Word(token[1], token[2], int(token[3]), token[4]))

        if len(sentence_data) > 0:
            data.append(sentence_data)

        return data

    def read_subword_data(self, subword_data_path):
        subword_data = list()
        sentence_data = list()
        sentence_count = 0

        with open(subword_data_path) as subword_file:
            for line in subword_file:
                if len(line) <= 1:
                    subword_data.append(sentence_data.copy())
                    sentence_data.clear()
                    sentence_count += 1
                elif line[0] != '#':
                    token = line.strip().split()
                    subword, word_idx = token[1], int(token[2])
                    assert word_idx <= len(self.data[sentence_count])
                    sentence_data.append(Subword(subword, word_idx))

        if len(sentence_data) > 0:
            subword_data.append(sentence_data)

        assert len(self.data) == len(subword_data)
        return subword_data
