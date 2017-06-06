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


def get_subword_windows(*args):
    input_sentence, input_subwords = args
    context_left = 2
    context_right = 2

    subwords = [subword_info.subword for subword_info in input_subwords]
    subword_windows = list()

    for subword_idx, subword in enumerate(subwords):
        subword_window = (['<S>'] * (context_left - subword_idx)) + subwords[subword_idx-context_left:subword_idx]
        right_boundary = subword_idx + context_right + 1
        subword_window += subwords[subword_idx:right_boundary] + (['</S>'] * (right_boundary - len(subwords)))
        subword_windows.append(subword_window)

    return subwords, subword_windows


def get_pos_labels(*args):
    input_sentence, _ = args
    return [word_info.pos for word_info in input_sentence]


def get_boundary_pos_labels(input_sentence, input_subwords):
    tags = list()
    for subword_idx, subword_info in enumerate(input_subwords):
        last_word_idx = input_subwords[subword_idx - 1].word_idx if subword_idx > 0 else None
        curr_word_idx = subword_info.word_idx
        next_word_idx = input_subwords[subword_idx + 1].word_idx if subword_idx + 1 < len(input_subwords) else None

        boundary = ''
        if last_word_idx != curr_word_idx != next_word_idx:
            boundary = 'S'
        elif last_word_idx != curr_word_idx and curr_word_idx == next_word_idx:
            boundary = 'B'
        elif last_word_idx == curr_word_idx == next_word_idx:
            boundary = 'I'
        elif last_word_idx == curr_word_idx and curr_word_idx != next_word_idx:
            boundary = 'E'

        tags.append(boundary + '-' + input_sentence[curr_word_idx-1].pos)

    return tags


def get_log_freq_labels(words, word_freq_map):
    return [word_freq_map.get(word, 0) for word in words]


def prepare_input_data(data_file, data_subword_file,
                       word_map, subword_map, pos_map, word_freq_map, split_func, pos_func, input_type):
    raw_data = ExperimentData(data_file, data_subword_file)

    x = dict()
    x['words_input'] = list()
    x['subwords_input'] = list()
    x['subwords_len'] = list()
    x['pos_labels'] = list()
    x['freq_labels'] = list()

    for sent_raw_words, sent_raw_subwords in zip(raw_data.data, raw_data.subword):
        split_words, split_subwords = split_func(sent_raw_words, sent_raw_subwords)
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

        sent_pos = [pos_map[tag] for tag in pos_func(sent_raw_words, sent_raw_subwords)]
        sent_freq = list()
        if input_type == 'word':
            sent_freq = get_log_freq_labels([word_info.word for word_info in sent_raw_words], word_freq_map)
        elif input_type == 'subword':
            sent_freq = get_log_freq_labels([subword_info.subword for subword_info in sent_raw_subwords], word_freq_map)

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
