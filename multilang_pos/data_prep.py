from tools.experiment_data import ExperimentData
from tools.batch_reader import BatchReader
import tools.data_transform as data_trans


def use_defined_subword(*args):
    input_sentence, input_subwords = args

    words = list()
    subwords = list()

    curr_subwords = list()
    latest_word_idx = 1
    for subword_info in input_subwords:
        if subword_info.word_idx != latest_word_idx:
            words.append(input_sentence[latest_word_idx - 1].word)
            subwords.append(curr_subwords)
            curr_subwords = list()

        curr_subwords.append(subword_info.subword)
        latest_word_idx = subword_info.word_idx

    words.append(input_sentence[latest_word_idx - 1].word)
    subwords.append(curr_subwords)

    return words, subwords


def prepare_input_data(data_file, data_subword_file, word_map, subword_map, pos_map, word_freq_map, split_func):
    raw_data = ExperimentData(data_file, data_subword_file)

    x = dict()
    x['words_input'] = list()
    x['subwords_input'] = list()
    x['subwords_len'] = list()
    x['pos_labels'] = list()
    x['freq_labels'] = list()

    for sent_words, sent_subwords in zip(raw_data.data, raw_data.subword):
        split_words, split_subwords = split_func(sent_words, sent_subwords)
        mapped_words = [word_map.get(word, word_map['<UNK>']) for word in split_words]

        x['words_input'].append(mapped_words)

        sent_subwords = list()
        sent_subwords_len = list()
        for word_subword in split_subwords:
            mapped_subwords = [subword_map.get(subword, subword_map['<UNK>']) for subword in word_subword]
            sent_subwords.append(mapped_subwords)
            sent_subwords_len.append(len(mapped_subwords))

        x['subwords_input'].append(sent_subwords)
        x['subwords_len'].append(sent_subwords_len)

        sent_pos = list()
        sent_freq = list()
        for word_info in sent_words:
            sent_pos.append(pos_map[word_info.pos])
            sent_freq.append(word_freq_map.get(word_info.word, 0))

        x['pos_labels'].append(sent_pos)
        x['freq_labels'].append(sent_freq)

    return x


def flatten_data(data):
    flatten_results = list()
    for sentence in data:
        flatten_results.extend(sentence)

    return flatten_results


def flatten_and_make_tensor(data):
    return data_trans.fill_right_to_tensor(flatten_data(data), 3)  # 3 = '<PAD>


def get_input_batch(input_dict, batch_size):
    data_list = [
        {'data': input_dict['words_input'], 'pad_element': None, 'post_func': None},
        {'data': input_dict['subwords_input'], 'pad_element': None, 'post_func': flatten_and_make_tensor},
        {'data': input_dict['subwords_len'], 'pad_element': None, 'post_func': flatten_data},
        {'data': input_dict['pos_labels'], 'pad_element': None, 'post_func': flatten_data},
        {'data': input_dict['freq_labels'], 'pad_element': None, 'post_func': flatten_data}
    ]

    input_feeder = BatchReader(data_list, batch_size)
    return input_feeder


def read_word_freq_map(freq_map_file_path):
    freq_map = dict()
    with open(freq_map_file_path) as freq_map_file:
        for line in freq_map_file:
            tokens = line.strip().split()
            freq_map[tokens[0]] = int(tokens[-1])  # token[0] = word, token[-1] = int(log_of_frequency)

    return freq_map
