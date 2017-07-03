from lstm_parser.data_prep import prepare_parser_data
from lstm_parser.parser_model import ParserModel
from lstm_parser.parser_action_model import ParserModel as ActionModel
from lstm_parser.parser_param_model import ParserModel as ParamModel
from lstm_parser.evaluate import get_epoch_evaluation
from lstm_parser.predict import predict_and_get_evaluation
from tools.embedding_reader import NetworkParams

from random import shuffle, uniform


def train(options):
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'word', 'action', 'bpos', 'pos', 'dep_label'])
    embeddings = {
        'word': network_params.params['word_embedding'],
        'subword': network_params.params['subword_embedding']
    }
    reverse_act_map = network_params.params['reverse_action_map']

    training_data_original = prepare_parser_data(options, network_params, 'train')
    eval_data = prepare_parser_data(options, network_params, 'eval')
    training_data = training_data_original.copy()

    model_list = ['action', 'param']
    model = dict()
    model['action'] = ActionModel(network_params.params, embeddings=embeddings)
    model['param'] = ParamModel(network_params.params, embeddings=embeddings)
    epoch_count = 0
    max_uas_score = 0
    train_loss = dict()
    uas_list = list()

    while True:
        print('Epoch', epoch_count)
        shuffle(training_data)
        all_epoch_loss = {key: 0 for key in model_list}
        for sent_idx, sentence in enumerate(training_data):
            for model_name in model_list:
                model[model_name].initial_parser_model(sentence['idx_subword'], sentence['idx_word_can'],
                                                  sentence['idx_bpos_can'],
                                                  sentence['only_subword'], sentence['buffer_packet'],
                                                  sentence['idx_buffer_packet'])

            all_parser_loss = {'action': 0, 'param': 0}
            for gold_action, feasible_action in zip(sentence['gold_actions'], sentence['feasible_actions']):
                for model_name in model_list:
                    train_loss[model_name] = model[model_name].calc_loss(gold_action, feasible_action)
                    all_parser_loss[model_name] += train_loss[model_name]
                    model[model_name].take_action(gold_action)

            for model_name in model_list:
                model[model_name].train()

                parser_loss = all_parser_loss[model_name] / len(sentence['gold_actions'])
                all_epoch_loss[model_name] += parser_loss
                if sent_idx % 50 == 0:
                    print('Parser ', sent_idx, ',',  model_name, 'loss = ', parser_loss)

        for model_name in model_list:
            epoch_loss = all_epoch_loss[model_name] / len(training_data)
            print('** Epoch', epoch_count, ',',  model_name, 'loss = ', epoch_loss)

        evaluation_list = list()
        for eval_sentence in eval_data:
            _, eval_dict = predict_and_get_evaluation(eval_sentence, model, network_params.params['action_map'], reverse_act_map)
            evaluation_list.append(eval_dict)

        epoch_eval = get_epoch_evaluation(evaluation_list)
        uas_score = epoch_eval['uas_f1_score']
        f1_score = epoch_eval['word_f1_score']
        print('f1 score = ', f1_score, ', uas score = ', uas_score)
        print(epoch_eval)

        if uas_score > max_uas_score:
            for model_name in model_list:
                model[model_name].save_model(options[model_name + '_model_save_path'], epoch_count)
            max_uas_score = max(max_uas_score, uas_score)

        uas_list.append(uas_score)
        average_uas = sum(uas_list) / len(uas_list)
        if uas_score < average_uas and epoch_count >= 20:
            print('Training completed.')
            break

        epoch_count += 1

