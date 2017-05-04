from collections import namedtuple

ParserInput = namedtuple('ParserInput', ['subword_idx', 'subword', 'word_candidates', 'pos_candidates'])


class InconsistentInputFile(Exception):
    def __init__(self, bad_file, sent_idx):
        Exception.__init__(self, 'Subword file and' + bad_file + ' is not consistent at sentence #'
                           + str(sent_idx))


class ParserInputData:
    def __init__(self, subword_file_path, word_seg_file_path, pos_file_path, n_word_seg=3, n_pos=3):
        self.data = ParserInputData.read_subword(subword_file_path)
        self.read_word_seg(word_seg_file_path, n_word_seg)
        self.read_pos_candidates(pos_file_path, n_pos)

    @staticmethod
    def read_subword(subword_file_path):
        subword_data = list()
        sentence_data = list()
        sentence_count = 0

        with open(subword_file_path) as subword_file:
            for line in subword_file:
                if len(line) <= 1:
                    subword_data.append(sentence_data.copy())
                    sentence_data.clear()
                    sentence_count += 1
                elif line[0] != '#':
                    token = line.strip().split()
                    subword_idx = int(token[0])
                    subword = token[1]
                    sentence_data.append(ParserInput(subword_idx, subword, [], []))

        if len(sentence_data) > 0:
            subword_data.append(sentence_data)

        return subword_data

    def read_word_seg(self, word_seg_file_path, n_word_seg):
        sentence_count = 0
        candidate_list = list()
        with open(word_seg_file_path) as word_seg_file:
            for line in word_seg_file:
                candidate_list.append(line.strip().split('  '))
                if len(candidate_list) == n_word_seg:
                    self.map_subword_with_candidate_words(sentence_count, candidate_list)
                    candidate_list = list()
                    sentence_count += 1

    def map_subword_with_candidate_words(self, sent_idx, candidate_list):
        position_list = [(0, 0)] * len(candidate_list)
        for subword_idx in range(len(self.data[sent_idx])):
            subword = self.data[sent_idx][subword_idx].subword
            new_position_list = list()

            for candidate, position in zip(candidate_list, position_list):
                (word_idx, char_idx) = position
                candidate_word = candidate[word_idx]
                end_idx = min(char_idx + len(subword), len(candidate_word))

                if subword != candidate_word[char_idx:end_idx]:
                    raise InconsistentInputFile('word segmentation file', sent_idx)

                self.data[sent_idx][subword_idx].word_candidates.append(candidate_word)

                # locate next word
                if end_idx == len(candidate_word):
                    new_position_list.append((word_idx+1, 0))
                else:
                    new_position_list.append((word_idx, end_idx))

            position_list = new_position_list

    def read_pos_candidates(self, pos_file_path, n_pos):
        sentence_count = 0
        sentence_info = list()
        with open(pos_file_path) as pos_file:
            for line in pos_file:
                if len(line) <= 1:
                    if len(sentence_info) != len(self.data[sentence_count]):
                        raise InconsistentInputFile('POS file', sentence_count)

                    for subword_idx in range(len(sentence_info)):
                        if self.data[sentence_count][subword_idx].subword != sentence_info[subword_idx][0]:
                            raise InconsistentInputFile('POS file', sentence_count)
                        self.data[sentence_count][subword_idx].pos_candidates.extend(sentence_info[subword_idx][1])

                    sentence_count += 1
                    sentence_info = list()

                elif line[0] != '#':
                    _, subword, pos, _ = line.strip().split('\t')
                    pos_list = pos.split(',')
                    if len(pos_list) != n_pos:
                        raise InconsistentInputFile('POS file', sentence_count)
                    sentence_info.append((subword, pos_list))

