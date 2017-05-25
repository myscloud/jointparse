import numpy as np


def read_index_file(index_file_path):
    map_dict = dict()
    reverse_map_dict = dict()
    with open(index_file_path) as index_file:
        for line in index_file:
            index_str, value = line.strip().split('\t')
            index = int(index_str)
            map_dict[value] = index
            reverse_map_dict[index] = value

    return map_dict, reverse_map_dict


def read_embedding_map(map_file_path):
    word_count = 0
    word_dict = dict()
    reverse_word_dict = dict()

    with open(map_file_path) as map_file:
        for line in map_file:
            word = line.strip()
            word_dict[word] = word_count
            reverse_word_dict[word_count] = word
            word_count += 1

    return word_dict, reverse_word_dict


def read_action_map(map_file_path):
    action_dict = dict()
    reverse_action_dict = dict()

    with open(map_file_path) as map_file:
        for line in map_file:
            index_str, action_set = line.strip().split('\t')
            index = int(index_str)
            action = tuple(action_set.split(' '))
            action_dict[action] = index
            reverse_action_dict[index] = action

    return action_dict, reverse_action_dict


class NetworkParams:
    def __init__(self):
        self.params = dict()

    def set_from_options(self, options, properties):
        if 'subword' in properties:
            self.set_subword_embedding(options['subword_embedding'], options['subword_embedding_map'])
        if 'word' in properties:
            self.set_word_embedding(options['word_embedding'], options['word_embedding_map'])
        if 'action' in properties:
            self.set_action_map(options['action_map'])
        if 'bpos' in properties:
            self.set_index_file('bpos_map', 'reverse_bpos_map', options['bpos_map'])
        if 'pos' in properties:
            self.set_index_file('pos_map', 'reverse_pos_map', options['pos_map'])
        if 'dep_label' in properties:
            self.set_index_file('dep_label_map', 'reverse_dep_label_map', options['dep_label_map'])

    def set_subword_embedding(self, embedding_file, embedding_map_file):
        self.params['subword_embedding'] = np.load(embedding_file)
        self.params['subword_map'], self.params['reverse_subword_map'] = read_embedding_map(embedding_map_file)

    def set_word_embedding(self, embedding_file, embedding_map_file):
        self.params['word_embedding'] = np.load(embedding_file)
        self.params['word_map'], self.params['reverse_word_map'] = read_embedding_map(embedding_map_file)

    def set_index_file(self, label, reverse_label, file_path):
        self.params[label], self.params[reverse_label] = read_index_file(file_path)

    def set_action_map(self, file_path):
        self.params['action_map'], self.params['reverse_action_map'] = read_action_map(file_path)

    def map_with_params(self, data_list, attribute_name, unk_name=None):
        """
        :param data_list: a 2D list of original data
        :param attribute_name: name of attribute to be mapped
        :param unk_name: name of a token's index to be replaced if the specific one isn't in the map
        :return: a 2D list of the same size with data_list, mapped with the attribute
        """
        assert attribute_name in self.params
        mapped_list = list()

        for sentence in data_list:
            curr_sentence = self.map_list_with_params(sentence, attribute_name, unk_name)
            mapped_list.append(curr_sentence)

        return mapped_list

    def map_list_with_params(self, data_list, attribute_name, unk_name=None):
        map_dict = self.params[attribute_name]
        mapped_list = list()
        for token in data_list:
            if token in map_dict:
                mapped_list.append(map_dict[token])
            elif unk_name:
                mapped_list.append(map_dict[unk_name])
            else:
                raise Exception

        return mapped_list

