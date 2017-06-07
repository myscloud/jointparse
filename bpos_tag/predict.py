from bpos_tag.tagger_model import TaggerModel
from bpos_tag.data_prep import read_word_freq_map, prepare_input_data, get_input_batch
import bpos_tag.data_prep as pos_data
from tools.embedding_reader import NetworkParams
from tools.experiment_data import ExperimentData

from numpy import argmax


def predict(options):
    # prepare data
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'word', 'pos', 'bpos'])
    # log_freq_map = read_word_freq_map(options['log_freq_file'])
    log_freq_map = read_word_freq_map(options['subword_log_freq_file'])

    data_types = ['train', 'eval', 'dev', 'test']
    model = TaggerModel(model_path=options['pos_model_load_path'])

    for data_type_name in data_types:
        print('Predicting', data_type_name, 'set.')

        data_file_path = options['ws_results_reformatted_path'] + 'data/' + data_type_name + '.ws'
        subword_file_path = options['ws_results_reformatted_path'] + 'subword/' + data_type_name + '.ws'

        raw_data = prepare_input_data(data_file_path, subword_file_path, network_params.params['subword_map'],
                                      network_params.params['subword_map'], network_params.params['bpos_map'],
                                      log_freq_map, pos_data.get_subword_windows, pos_data.get_boundary_pos_labels,
                                      'subword')
        # raw_data = prepare_input_data(data_file_path, subword_file_path, network_params.params['word_map'], network_params.params['subword_map'], network_params.params['pos_map'], log_freq_map, pos_data.use_defined_subword)
        exp_data = ExperimentData(data_file_path, subword_file_path)

        data_feeder = get_input_batch(raw_data, 1)

        sent_count = 0
        results = list()
        while not data_feeder.is_epoch_end():
            batch_input = data_feeder.get_next_batch()
            input_list = batch_input[0:3]
            predicted_tags = model.predict(input_list)
            # words = [word_info.word for word_info in exp_data.data[sent_count]]
            words = [subword_info.subword for subword_info in exp_data.subword[sent_count]]

            results.append([])
            for word, tag_score in zip(words, predicted_tags):
                # decoded_tag = network_params.params['reverse_pos_map'][argmax(tag_score)]
                decoded_tag = network_params.params['reverse_bpos_map'][argmax(tag_score)]
                results[-1].append((word, decoded_tag))

            sent_count += 1

        output_file = options['pos_results_path'] + data_type_name + '.ws'
        with open(output_file, 'w') as out_file:
            for sent_idx, sentence in enumerate(results):
                out_file.write('#Sentence ' + str(sent_idx) + '\n')
                for word_idx, (word, tag) in enumerate(sentence):
                    out_file.write(str(word_idx + 1) + '\t' + word + '\t' + tag + '\t0\t<PAD>\n')
                out_file.write('\n')


