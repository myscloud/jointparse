from collections import defaultdict

from parse.parser import Parser
from parse.parser_action import get_feasible_actions, take_parser_action_from_index
import tools.data_transform as data_trans
import numpy as np


def predict(batch_parsers, model, feature_pattern, action_list, reverse_action_map):
    batch_size = len(batch_parsers)
    n_actions = len(reverse_action_map)
    n_terminated_parsers = 0
    feature_dict = feature_pattern.feature_dict

    # if some fields is not a parser, consider it as a terminated parser
    for parser in batch_parsers:
        if not isinstance(parser, Parser):
            n_terminated_parsers += 1

    iteration = 0
    while True:
        input_index_dict = defaultdict(list)
        feasible_actions = list()

        for parser in batch_parsers:
            if isinstance(parser, Parser) and not parser.is_terminated():
                features = parser.get_features(feature_pattern)
                parser_feasible_act = get_feasible_actions(parser, action_list)
                feasible_actions.append(parser_feasible_act)
            else:
                features = None
                feasible_actions.append([0] * n_actions)

            for category in feature_dict:
                if features is not None:
                    input_index_dict[category].append(features[category])
                else:
                    input_index_dict[category].append([0] * len(feature_dict[category]))

        input_dict = dict()
        for category in input_index_dict:
            input_dict[category] = data_trans.to_matrix((input_index_dict[category]))

        feasible_action_matrix = data_trans.to_matrix(feasible_actions)

        predicted_action_matrix = model.predict(input_dict, feasible_action_matrix)
        for parser, predicted_action in zip(batch_parsers, predicted_action_matrix):
            if isinstance(parser, Parser) and not parser.is_terminated():
                action_index = np.argmax(predicted_action)
                take_parser_action_from_index(parser, action_index, reverse_action_map, action_list)
                if parser.is_terminated():
                    n_terminated_parsers += 1

        if n_terminated_parsers == batch_size:
            break

        iteration += 1

    return batch_parsers
