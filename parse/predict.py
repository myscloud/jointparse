from parse.parser_action import get_feasible_actions, take_parser_action_from_index
import tools.data_transform as data_trans
import numpy as np

from parse.parser_model import ParserModel
from parse.feature_pattern import FeaturePattern
from parse.parser import Parser
from parse.parser_action import get_action_index
from parse.evaluate import get_parser_evaluation, get_epoch_evaluation

from tools.parser_input_data import ParserInputData
from tools.experiment_data import ExperimentData
from tools.embedding_reader import NetworkParams


def predict(parser, model, feature_pattern, action_list, reverse_action_map):
    iteration = 0
    while not parser.is_terminated():
        features = parser.get_features(feature_pattern)
        parser_feasible_act = get_feasible_actions(parser, action_list)

        input_dict = dict()
        for category in features:
            input_dict[category] = data_trans.to_matrix([features[category]])

        feasible_action_matrix = data_trans.to_matrix([parser_feasible_act])

        predicted_action_matrix = model.predict(input_dict, feasible_action_matrix)
        predicted_action = predicted_action_matrix[0]
        action_index = np.argmax(predicted_action)
        take_parser_action_from_index(parser, action_index, reverse_action_map, action_list)

        iteration += 1

    return parser


def prepare_input(options):
    training_parser_input = ParserInputData(options['train_subword_file_path'], options['train_word_seg_path'],
                                         options['train_pos_candidates_path'])

    training_parser_label = ExperimentData(options['train_file_path'], options['train_subword_file_path'])

    eval_parser_input = ParserInputData(options['eval_subword_file_path'], options['eval_word_seg_path'],
                                        options['eval_pos_candidates_path'])
    eval_parser_label = ExperimentData(options['eval_file_path'], options['eval_subword_file_path'])
    dev_parser_input = ParserInputData(options['dev_subword_file_path'], options['dev_word_seg_path'],
                                       options['dev_pos_candidates_path'])
    dev_parser_label = ExperimentData(options['dev_file_path'], options['dev_subword_file_path'])
    test_parser_input = ParserInputData(options['test_subword_file_path'], options['test_word_seg_path'],
                                        options['test_pos_candidates_path'])
    test_parser_label = ExperimentData(options['test_file_path'], options['test_subword_file_path'])

    network_params = NetworkParams()
    network_params.set_subword_embedding(options['subword_embedding'], options['subword_embedding_map'])
    network_params.set_word_embedding(options['word_embedding'], options['word_embedding_map'])
    network_params.set_action_map(options['action_map'])
    network_params.set_index_file('bpos_map', 'reverse_bpos_map', options['bpos_map'])
    network_params.set_index_file('pos_map', 'reverse_pos_map', options['pos_map'])
    network_params.set_index_file('dep_label_map', 'reverse_dep_label_map', options['dep_label_map'])

    feature_pattern = FeaturePattern(options['feature_pattern_file'], network_params=network_params)

    parser_input = {'training': training_parser_input, 'eval': eval_parser_input,
                    'dev': dev_parser_input, 'test': test_parser_input}
    parser_label = {'training': training_parser_label, 'eval': eval_parser_label,
                    'dev': dev_parser_label, 'test': test_parser_label}
    return parser_input, parser_label, network_params, feature_pattern


def write_log(log_path, epoch_count, batch_loss, evaluation=None):
    log_file_path = log_path + 'epoch-' + '{0:04}'.format(epoch_count)
    with open(log_file_path, 'w') as log_file:
        log_file.write('epoch ' + str(epoch_count) + '\n')
        log_file.write(str(batch_loss) + '\n')
        if evaluation is not None:
            log_file.write(str(evaluation) + '\n')


def predict_data_set(options):
    # data_set_name = ['training', 'eval', 'dev', 'test']
    data_set_name = ['eval']
    parser_input, parser_label, network_params, feature_pattern = prepare_input(options)

    input_embedding = {
        'word': network_params.params['word_embedding'],
        'subword': network_params.params['subword_embedding']
    }

    action_list = get_action_index(network_params.params['action_map'])
    model = ParserModel(input_embedding, model_path=options['parser_model_load_path'])

    for data_set in data_set_name:
        evaluation_list = list()
        parser_count = 0

        print(data_set)
        for sentence, data_label, subword_label in zip(parser_input[data_set].data,
                                                       parser_label[data_set].data, parser_label[data_set].subword):
            parser = Parser(sentence, network_params, labels=[data_label, subword_label])
            predicted_parser = predict(parser, model, feature_pattern,
                                       action_list, network_params.params['reverse_action_map'])
            evaluation = get_parser_evaluation(predicted_parser.results[1:], data_label)
            evaluation_list.append(evaluation)
            parser_count += 1
            if parser_count % 10 == 0:
                print('.')

        epoch_eval = get_epoch_evaluation(evaluation_list)
        print(epoch_eval)
        print('-----------------------')
