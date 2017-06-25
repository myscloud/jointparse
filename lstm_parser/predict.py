from lstm_parser.evaluate import get_parser_evaluation
from lstm_parser.data_prep import get_feasible_action_index
from lstm_parser.parser import Parser


def predict_and_get_evaluation(sentence, model, reverse_action_map):
    parser = Parser(sentence['sentence_data'])
    model.initial_parser_model(sentence['idx_subword'], sentence['idx_word_can'], sentence['idx_bpos_can'],
                               sentence['only_subword'], sentence['buffer_packet'], sentence['idx_buffer_packet'])
    while not parser.is_parsing_terminated():
        feasible_actions = get_feasible_action_index(parser.get_feasible_actions(), reverse_action_map)
        action_index = model.predict(feasible_actions)
        action_tuple = reverse_action_map[action_index]
        parser.take_action(action_tuple)
        model.take_action(action_index)

    eval_dict = get_parser_evaluation(parser.results[1:], sentence['gold_data'])
    return parser.results[1:], eval_dict
