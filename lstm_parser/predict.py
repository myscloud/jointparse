from lstm_parser.evaluate import get_parser_evaluation, get_epoch_evaluation
from lstm_parser.data_prep import get_feasible_action_index, prepare_parser_data
from lstm_parser.parser import Parser
from lstm_parser.parser_action_model import ParserModel as ActionModel
from lstm_parser.parser_param_model import ParserModel as ParamModel
from lstm_parser.parser_model import ParserModel

from tools.embedding_reader import NetworkParams


def predict_and_get_evaluation(sentence, models, action_map, reverse_action_map):
    parser = Parser(sentence['sentence_data'])
    for model_name in models:
        models[model_name].initial_parser_model(sentence['idx_subword'], sentence['idx_word_can'], sentence['idx_bpos_can'],
                                   sentence['only_subword'], sentence['buffer_packet'], sentence['idx_buffer_packet'])

    while not parser.is_parsing_terminated():
        feasible_actions = get_feasible_action_index(parser.get_feasible_actions(), reverse_action_map)
        action = models['action'].predict(feasible_actions)
        pos, rel = models['param'].predict(feasible_actions)
        param = rel if (action == 'LEFT-ARC' or action == 'RIGHT-ARC') else pos
        action_index = action_map[(action, param)]
        action_tuple = reverse_action_map[action_index]
        parser.take_action(action_tuple)

        for model_name in models:
            models[model_name].take_action(action_index)

    eval_dict = get_parser_evaluation(parser.results[1:], sentence['gold_data'])
    return parser.results[1:], eval_dict


def predict(options):
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'word', 'action', 'bpos', 'pos', 'dep_label'])
    embeddings = {
        'word': network_params.params['word_embedding'],
        'subword': network_params.params['subword_embedding']
    }
    action_map = network_params.params['action_map']
    reverse_act_map = network_params.params['reverse_action_map']

    model = dict()
    model['action'] = ActionModel(network_params.params, embeddings=embeddings, model_path=options['action_model_load_path'])
    model['param'] = ParamModel(network_params.params, embeddings=embeddings, model_path=options['param_model_load_path'])
    # data_set = ['train', 'eval', 'dev', 'test']
    data_set = ['test']
    for data_set_name in data_set:
        print('Evaluating ', data_set_name, 'set.')
        data = prepare_parser_data(options, network_params, data_set_name)

        evaluation_list = list()
        for sent_idx, sentence in enumerate(data):
            _, eval_dict = predict_and_get_evaluation(sentence, model, action_map, reverse_act_map)
            evaluation_list.append(eval_dict)
            if sent_idx % 50 == 0:
                print('Evaluating sentence', sent_idx)

        epoch_eval = get_epoch_evaluation(evaluation_list)

        print(data_set_name)
        print(epoch_eval)
