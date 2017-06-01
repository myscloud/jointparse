from tools.experiment_data import ExperimentData
from tools.batch_reader import BatchReader
import tools.data_transform as data_trans


def get_input_batch(data_file, subword_data_file, network_params, options):
    exp_data = ExperimentData(data_file, subword_data_file)
    subwords, labels, bigrams, second_labels = prepare_input(exp_data, network_params, options)

    data_list = [
        {'data': subwords, 'pad_element': None, 'post_func': data_trans.to_matrix},
        {'data': bigrams, 'pad_element': None, 'post_func': data_trans.to_matrix},
        {'data': labels, 'pad_element': None, 'post_func': data_trans.to_flatten_tensor},
        {'data': second_labels, 'pad_element': None, 'post_func': data_trans.to_flatten_tensor}
    ]
    data_feeder = BatchReader(data_list, 1)
    return data_feeder, exp_data


def prepare_input(exp_data, network_params, options):
    left_pad = network_params.params['subword_map']['<S>']
    right_pad = network_params.params['subword_map']['</S>']
    bigram_map = network_params.params['bigram_map']

    subwords = list()
    labels = list()
    all_second_labels = list()
    bigrams = list()

    for sentence in exp_data.subword:
        sent_subwords, sent_labels = get_labels(sentence)
        mapped_labels = network_params.map_list_with_params(sent_labels, 'ws_label')
        second_labels = get_second_labels(mapped_labels, 2, 2)
        mapped_subwords = network_params.map_list_with_params(sent_subwords, 'subword_map', unk_name='<UNK>')
        subword_windows = get_context_windows(mapped_subwords, options['no_context_left'], options['no_context_right'],
                                              left_pad, right_pad)
        bigram_windows = get_bigram_windows(subword_windows, bigram_map)

        subwords.append(subword_windows)
        labels.append(mapped_labels)
        all_second_labels.append(second_labels)
        bigrams.append(bigram_windows)

    return subwords, labels, bigrams, all_second_labels


def get_labels(sentence):
    subword_list = list()
    label_list = list()

    for subword_idx, subword_info in enumerate(sentence):
        subword_list.append(subword_info.subword)

        prev_word_idx = sentence[subword_idx-1].word_idx if subword_idx > 0 else -1
        curr_word_idx = sentence[subword_idx].word_idx
        next_word_idx = sentence[subword_idx+1].word_idx if subword_idx + 1 < len(sentence) else -1

        if prev_word_idx != curr_word_idx and curr_word_idx != next_word_idx:
            label_list.append('S')
        elif prev_word_idx != curr_word_idx and curr_word_idx == next_word_idx:
            label_list.append('B')
        elif prev_word_idx == curr_word_idx and curr_word_idx == next_word_idx:
            label_list.append('I')
        elif prev_word_idx == curr_word_idx and curr_word_idx != next_word_idx:
            label_list.append('E')

    return subword_list, label_list


def get_second_labels(labels, n_left, n_right):
    padded_labels = ([3] * n_left) + labels + ([3] * n_right)
    second_labels = list()
    for label_idx in range(n_left, len(padded_labels)-n_right):
        labels_window = padded_labels[label_idx-n_left:label_idx+n_right+1]
        label_sum = 0
        for i in range(1, len(labels_window)):
            label_sum *= 2
            if labels_window[i-1] < labels_window[i] != 3:
                label_sum += 1

        second_labels.append(label_sum)

    return second_labels


def get_context_windows(sentence, n_left, n_right, left_pad, right_pad):
    window_list = list()
    for subword_idx in range(len(sentence)):
        window = list()
        if subword_idx < n_left:
            window += ([left_pad] * (n_left - subword_idx))
        window += sentence[max(0, subword_idx-n_left):subword_idx]
        window += [sentence[subword_idx]]
        window += sentence[subword_idx+1:min(len(sentence), subword_idx+n_right+1)]
        if subword_idx + n_right >= len(sentence):
            window += ([right_pad] * (n_right - (len(sentence) - subword_idx) + 1))

        window_list.append(window)

    return window_list


def get_bigram_windows(subword_windows, bigram_map):
    window_list = list()
    for subword_window in subword_windows:
        window = list()
        for subword_idx in range(1, len(subword_window)):
            bigram = (subword_window[subword_idx-1], subword_window[subword_idx])
            bigram_idx = bigram_map.get(bigram, 0)
            window.append(bigram_idx)
        window_list.append(window)

    return window_list


def read_transition_prob(transition_file_path):
    with open(transition_file_path) as transition_file:
        lines = transition_file.readlines()

    transition_prob = list()
    all_count = int(lines[0].strip())
    for line in lines[1:]:
        tag_prob = list()
        tokens = line.strip().split('\t')
        for token in tokens:
            tag_prob.append(int(token) / all_count)
        transition_prob.append(tag_prob)

    return transition_prob


def write_results(output_file_path, subwords, labels):
    label_map = ['B', 'I', 'E', 'S']
    with open(output_file_path, 'w') as out_file:
        for sent_subwords, candidates in zip(subwords, labels):
            for candidate_labels in candidates:
                segmented_words = get_segmented_words_from_labels(sent_subwords, candidate_labels)
                out_file.write(' '.join(segmented_words) + '\n')
                out_file.write(' '.join([label_map[label] for label in candidate_labels]) + '\n')


def get_segmented_words_from_labels(subwords, labels):
    words = list()
    curr_word = ''
    for subword, label in zip(subwords, labels):
        if label == 0 or label == 3:  # B or S
            if curr_word != '':
                words.append(curr_word)
            curr_word = subword
        else:
            curr_word += subword

        if label == 2 or label == 3:  # E or S
            words.append(curr_word)
            curr_word = ''

    if curr_word != '':
        words.append(curr_word)
    return words
