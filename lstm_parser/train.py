from lstm_parser.data_prep import prepare_parser_data
from lstm_parser.parser_model import ParserModel
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

    model = ParserModel(network_params.params, embeddings=embeddings)
    epoch_count = 0

    while True:
        print('Epoch', epoch_count)
        shuffle(training_data)
        all_epoch_loss = 0
        for sent_idx, sentence in enumerate(training_data):
            model.initial_parser_model(sentence['idx_subword'], sentence['idx_word_can'], sentence['idx_bpos_can'],
                                       sentence['only_subword'])

            all_parser_loss = 0
            for gold_action, feasible_action in zip(sentence['gold_actions'], sentence['feasible_actions']):
                train_loss = model.calc_loss(gold_action, feasible_action)
                print(train_loss)
                all_parser_loss += train_loss
                model.take_action(gold_action)

            model.train()
            parser_loss = all_parser_loss / len(sentence['gold_actions'])
            all_epoch_loss += parser_loss
            if sent_idx % 1 == 0:
                print('Parser ', sent_idx, ', loss = ', parser_loss)

            if sent_idx > 3:
                break

            # if sent_idx % 5 == 0:
            #     number = int(uniform(0, len(eval_data)-1))
            #     results, eval_dict = predict_and_get_evaluation(eval_data[number], model, reverse_act_map)
            #     print('-- eval#', number, ', f1 score =', eval_dict['f1_score'], ', uas =', eval_dict['uas_score'])
            #     print(results)

        epoch_loss = all_epoch_loss / len(training_data)
        print('***** Epoch', epoch_count, 'loss = ', epoch_loss)

        epoch_count += 1
        break

