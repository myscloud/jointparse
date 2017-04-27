from tools.parser_input_data import ParserInputData
from tools.embedding_reader import NetworkParams

from parse.parser import Parser
from parse.feature_pattern import FeaturePattern


def prepare_input(options):
    train_parser_input = ParserInputData(options['dev_subword_file_path'], options['dev_word_seg_path'],
                                         options['dev_pos_candidates_path'])
    network_params = NetworkParams()
    network_params.set_subword_embedding(options['subword_embedding'], options['subword_embedding_map'])
    network_params.set_word_embedding(options['word_embedding'], options['word_embedding_map'])
    network_params.set_action_map(options['action_map'])
    network_params.set_index_file('bpos_map', 'reverse_bpos_map', options['bpos_map'])
    network_params.set_index_file('pos_map', 'reverse_pos_map', options['pos_map'])
    network_params.set_index_file('dep_label_map', 'reverse_dep_label_map', options['dep_label_map'])

    feature_pattern = FeaturePattern(options['feature_pattern_file'], network_params=network_params)

    return train_parser_input, network_params, feature_pattern


def train(options):
    train_parser_input, network_params, feature_pattern = prepare_input(options)
    for sentence in train_parser_input.data:
        parser = Parser(sentence, network_params)
        print(parser.get_features(feature_pattern))
        break
