import numpy as np

from tools.experiment_data import ExperimentData
from tools.embedding_reader import NetworkParams
from word_seg.data_prep import prepare_input


def prepare_bigram_embedding(options):
    data = ExperimentData(options['train_file_path'], options['train_subword_file_path'])
    bigram_set = set()

    network_params = NetworkParams()
    network_params.set_subword_embedding(options['subword_embedding'], options['subword_embedding_map'])
    embedding = network_params.params['subword_embedding']
    subword_map = network_params.params['subword_map']

    sent_start_index = subword_map['<S>']
    sent_end_index = subword_map['</S>']
    unk_index = subword_map['<UNK>']

    for sentence in data.subword:
        for subword_idx in range(1, len(sentence)):
            first_subword_index = subword_map.get(sentence[subword_idx-1].subword, unk_index)
            last_subword_index = subword_map.get(sentence[subword_idx].subword, unk_index)
            bigram = (first_subword_index, last_subword_index)
            bigram_set.add(bigram)
        bigram_set.add((sent_start_index, subword_map.get(sentence[0].subword, unk_index)))
        bigram_set.add((subword_map.get(sentence[-1].subword, unk_index), sent_end_index))

    bigram_set.add((sent_start_index, sent_start_index))
    bigram_set.add((sent_end_index, sent_end_index))

    bigram_list = sorted(list(bigram_set))
    bigram_emb = list()
    for (first_subword, last_subword) in bigram_list:
        new_emb = (embedding[first_subword, :] + embedding[last_subword, :]) / 2
        bigram_emb.append(new_emb)

    bigram_embeddings = np.stack(tuple(bigram_emb))

    # save
    with open(options['bigram_embedding_map'], 'w') as emb_map_file:
        for (first_index, last_index) in bigram_list:
            emb_map_file.write(str(first_index) + '\t' + str(last_index) + '\n')
    np.save(options['bigram_embedding'], bigram_embeddings)
    print(bigram_embeddings.shape)


def prepare_transition_prob(options):
    data = ExperimentData(options['train_file_path'], options['train_subword_file_path'])
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'ws_label', 'subword_bigram'])

    _, labels, _ = prepare_input(data, network_params, options)
    no_labels = len(network_params.params['ws_label'])
    transition_count = list()
    for _ in range(no_labels):
        transition_count.append([0] * no_labels)
    all_count = 0

    for sentence in labels:
        for label_idx in range(1, len(sentence)):
            first_label = sentence[label_idx-1]
            second_label = sentence[label_idx]
            transition_count[first_label][second_label] += 1
            all_count += 1

    with open(options['transition_prob_file'], 'w') as prob_file:
        prob_file.write(str(all_count) + '\n')
        for n_label in range(len(transition_count)):
            prob_file.write('\t'.join([str(count_number) for count_number in transition_count[n_label]]) + '\n')
