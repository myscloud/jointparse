from collections import Counter
from math import log2

from tools.experiment_data import ExperimentData
from tools.embedding_reader import NetworkParams


def get_log_freq(options):
    training_data = ExperimentData(options['train_file_path'], options['train_subword_file_path'])
    network_params = NetworkParams()
    network_params.set_word_embedding(options['word_embedding'], options['word_embedding_map'])

    words = [word_info.word for sent in training_data.data for word_info in sent]
    frequencies = Counter(words)
    frequency_list = list()
    unk_count = 0
    for word, freq in list(frequencies.items()):
        if freq <= 1:
            unk_count += 1
        else:
            frequency_list.append((word, freq))
    frequency_list.append(('<UNK>', unk_count))

    sorted_frequencies = sorted(frequency_list, key=lambda x: x[1], reverse=True)

    with open(options['log_freq_file'], 'w') as freq_log_file:
        for word, freq in sorted_frequencies:
            freq_log = log2(freq)
            int_freq_log = int(freq_log)
            freq_log_file.write(word + '\t' + str(freq) + '\t' + str(freq_log) + '\t' + str(int_freq_log) + '\n')


def get_subword_log_freq(options):
    training_data = ExperimentData(options['train_file_path'], options['train_subword_file_path'])
    network_params = NetworkParams()
    network_params.set_subword_embedding(options['subword_embedding'], options['subword_embedding_map'])

    subwords = [subword_info.subword for sent in training_data.subword for subword_info in sent]
    frequencies = Counter(subwords)
    frequency_list = list()
    unk_count = 0
    for word, freq in list(frequencies.items()):
        if freq <= 1:
            unk_count += 1
        else:
            frequency_list.append((word, freq))
    frequency_list.append(('<UNK>', unk_count))

    sorted_frequencies = sorted(frequency_list, key=lambda x: x[1], reverse=True)

    with open(options['subword_log_freq_file'], 'w') as freq_log_file:
        for subword, freq in sorted_frequencies:
            freq_log = log2(freq)
            int_freq_log = int(freq_log)
            freq_log_file.write(subword + '\t' + str(freq) + '\t' + str(freq_log) + '\t' + str(int_freq_log) + '\n')
