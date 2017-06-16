from lstm_parser.data_prep import prepare_parser_data, get_feasible_action_index
from lstm_parser.parser_model import ParserModel
from lstm_parser.parser import Parser
from lstm_parser.evaluate import get_parser_evaluation, get_epoch_evaluation
from tools.embedding_reader import NetworkParams

from random import shuffle


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

    model = ParserModel(network_params.params, embeddings=embeddings)
    epoch_count = 0
    max_uas = 0
    uas_list = list()

    while True:
        print('Epoch', epoch_count)
        shuffle(training_data)
        all_epoch_loss = 0
        for sent_idx, sentence in enumerate(training_data):
            model.initial_parser_model(sentence['idx_subword'], sentence['idx_word_can'], sentence['idx_bpos_can'],
                                       sentence['only_subword'])

            all_parser_loss = 0
            for gold_action, feasible_action in zip(sentence['gold_actions'], sentence['feasible_actions']):
                train_loss = model.train(gold_action, feasible_action)
                all_parser_loss += train_loss
                model.take_action(gold_action)

            parser_loss = all_epoch_loss / len(sentence['gold_actions'])
            all_epoch_loss += parser_loss
            if sent_idx % 10 == 0:
                print('Parser ', sent_idx, ', loss = ', parser_loss)

        epoch_loss = all_epoch_loss / len(training_data)
        print('-- Epoch', epoch_count, 'loss = ', epoch_loss)

        # evaluate
        print('Evaluate...')
        evaluation_list = list()
        for sent_idx, sentence in enumerate(eval_data):
            model.initial_parser_model(sentence['idx_subword'], sentence['idx_word_can'], sentence['idx_bpos_can'],
                                       sentence['only_subword'])
            parser = Parser(sentence['sentence_data'])
            while not parser.is_parsing_terminated():
                feasible_action = get_feasible_action_index(parser.get_feasible_actions(), reverse_act_map)
                action_index = model.predict(feasible_action)
                action_tuple = reverse_act_map[action_index]
                parser.take_action(action_tuple)
                model.take_action(action_index)

            eval_dict = get_parser_evaluation(parser.results[1:], sentence['gold_data'])
            if sent_idx % 10 == 0:
                print('Sentence', sent_idx, ', f1_score =', eval_dict['f1_score'], ', uas_score = ', eval_dict['uas_score'])
            evaluation_list.append(eval_dict)

        epoch_eval = get_epoch_evaluation(evaluation_list)
        print('=== Epoch Evaluation:')
        print(epoch_eval)

        uas_score = epoch_eval['uas_score']
        uas_list.append(uas_score)
        average_uas = sum(uas_list) / len(uas_list)

        if uas_score > max_uas:
            max_uas = uas_score
            model.save_model(options['parser_model_save_path'], epoch_count)

        if uas_score < average_uas and epoch_count >= 10:
            break

        epoch_count += 1

