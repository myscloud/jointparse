from parse.parser_action import get_feasible_actions, take_parser_action_from_index
import tools.data_transform as data_trans
import numpy as np


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
