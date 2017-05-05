from collections import deque

from tools.parser_input_data import ParserInputData
from tools.experiment_data import ExperimentData
from tools.embedding_reader import NetworkParams
import tools.data_transform as data_trans
from tools.batch_reader import BatchReader

from parse.parser import Parser
from parse.parser_action import take_parser_action, get_feasible_actions, get_action_index
from parse.feature_pattern import FeaturePattern
from parse.parser_model import ParserModel
from parse.predict import predict
from parse.evaluate import get_parser_evaluation, get_epoch_evaluation


def prepare_input(options):
    training_parser_input = ParserInputData(options['train_subword_file_path'], options['train_word_seg_path'],
                                         options['train_pos_candidates_path'])

    training_parser_label = ExperimentData(options['train_file_path'], options['train_subword_file_path'])

    eval_parser_input = ParserInputData(options['eval_subword_file_path'], options['eval_word_seg_path'],
                                        options['eval_pos_candidates_path'])
    eval_parser_label = ExperimentData(options['eval_file_path'], options['eval_subword_file_path'])

    network_params = NetworkParams()
    network_params.set_subword_embedding(options['subword_embedding'], options['subword_embedding_map'])
    network_params.set_word_embedding(options['word_embedding'], options['word_embedding_map'])
    network_params.set_action_map(options['action_map'])
    network_params.set_index_file('bpos_map', 'reverse_bpos_map', options['bpos_map'])
    network_params.set_index_file('pos_map', 'reverse_pos_map', options['pos_map'])
    network_params.set_index_file('dep_label_map', 'reverse_dep_label_map', options['dep_label_map'])

    feature_pattern = FeaturePattern(options['feature_pattern_file'], network_params=network_params)

    return training_parser_input, training_parser_label, eval_parser_input, eval_parser_label,\
           network_params, feature_pattern


def train(options):
    training_parser_input, training_parser_label, eval_parser_input, eval_parser_label\
        , network_params, feature_pattern = prepare_input(options)
    feature_categories = list(feature_pattern.feature_dict.keys())
    action_list = get_action_index(network_params.params['action_map'])

    # prepare training pairs; a list of (input, label)
    training_pairs = {'x': {}, 'y': [], 'feasible_actions': []}
    for feature in feature_categories:
        training_pairs['x'][feature] = list()

    for sentence, data_label, subword_label in \
            zip(training_parser_input.data, training_parser_label.data, training_parser_label.subword):
        parser = Parser(sentence, network_params, labels=[data_label, subword_label])
        while not parser.is_terminated():
            x_features = parser.get_features(feature_pattern)
            for feature in x_features:  # e.g. subword, word, pos, ...
                training_pairs['x'][feature].append(x_features[feature])

            action = parser.get_next_gold_standard_action()
            action_label = network_params.params['action_map'][action]
            training_pairs['y'].append(action_label)

            training_pairs['feasible_actions'].append(get_feasible_actions(parser, action_list))
            take_parser_action(parser, action)

    # start training...
    input_embedding = {
        'word': network_params.params['word_embedding'],
        'subword': network_params.params['subword_embedding']
    }
    model = ParserModel(input_embedding)

    training_input_list = list()
    for feature in feature_categories:
        training_input_list.append(
            {'data': training_pairs['x'][feature], 'pad_element': None, 'post_func': data_trans.to_matrix})
    training_input_list.append(
        {'data': training_pairs['y'], 'pad_element': None, 'post_func': data_trans.to_matrix})
    training_input_list.append(
        {'data': training_pairs['feasible_actions'], 'pad_element': None, 'post_func': data_trans.to_matrix}
    )

    batch_size = options['parsing_batch_size']
    training_input_feeder = BatchReader(training_input_list, batch_size)
    epoch_count = 0
    last_f1 = deque(maxlen=20)
    last_uas = deque(maxlen=20)
    last_batch_loss = deque(maxlen=20)
    max_f1 = 0
    max_uas = 0

    while True:
        print('epoch', epoch_count)
        iteration_count = 0
        training_loss_sum = 0
        training_input_feeder.shuffle()

        # training
        while not training_input_feeder.is_epoch_end():
            batch_training_pairs = training_input_feeder.get_next_batch()
            input_dict = dict()
            for feature_idx, feature in enumerate(feature_categories):
                input_dict[feature] = batch_training_pairs[feature_idx]
            labels = batch_training_pairs[-2]
            feasible_actions = batch_training_pairs[-1]
            loss = model.train(input_dict, labels, feasible_actions)

            training_loss_sum += loss
            iteration_count += 1

        training_batch_loss = training_loss_sum / iteration_count
        last_batch_loss.append(training_batch_loss)
        print(training_batch_loss)

        # evaluate
        parsers = list()
        evaluation_list = list()

        for sentence, data_label, subword_label in zip(
                eval_parser_input.data, eval_parser_label.data, eval_parser_label.subword):
            parser = Parser(sentence, network_params, labels=[data_label, subword_label])
            parsers.append(parser)

        parser_list = [{'data': parsers, 'pad_element': [None], 'post_func': None},
                       {'data': eval_parser_label.data, 'pad_element': [None], 'post_func': None}]

        parser_feeder = BatchReader(parser_list, batch_size)

        while not parser_feeder.is_epoch_end():
            batch_parsers, batch_gold_data = parser_feeder.get_next_batch()
            predicted_parsers = predict(batch_parsers, model, feature_pattern,
                                        action_list, network_params.params['reverse_action_map'])

            for predicted_parser, gold_results in zip(predicted_parsers, batch_gold_data):
                if isinstance(predicted_parser, Parser):
                    evaluation = get_parser_evaluation(predicted_parser.results[1:], gold_results)
                    evaluation_list.append(evaluation)

        epoch_eval = get_epoch_evaluation(evaluation_list)
        f1_score = epoch_eval['word_f1_score']
        uas_score = epoch_eval['uas_accuracy']
        last_f1.append(f1_score)
        last_uas.append(uas_score)
        max_f1 = max(f1_score, max_f1)
        max_uas = max(uas_score, max_uas)

        average_f1 = sum(last_f1) / len(last_f1)
        average_uas = sum(last_uas) / len(last_uas)
        average_batch_loss = sum(last_batch_loss) / len(last_batch_loss)
        print(f1_score, uas_score)

        if training_batch_loss < average_batch_loss or f1_score > average_f1 \
                or uas_score > average_uas or epoch_count < 20:
            pass
        else:
            model.save_model(options['parser_model_save_path'], epoch_count)
            break

        epoch_count += 1
