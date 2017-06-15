from lstm_parser.data_prep import prepare_parser_data
from lstm_parser.parser_model import ParserModel
from tools.embedding_reader import NetworkParams

from random import shuffle

def train(options):
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'word', 'action', 'bpos', 'pos', 'dep_label'])
    embeddings = {
        'word': network_params.params['word_embedding'],
        'subword': network_params.params['subword_embedding']
    }

    training_data_original = prepare_parser_data(options, network_params, 'train')
    eval_data = prepare_parser_data(options, network_params, 'eval')
    training_data = training_data_original.copy()

    model = ParserModel(network_params.params, embeddings=embeddings)
    epoch_count = 0

    while True:
        print('Epoch', epoch_count)
        shuffle(training_data)
        all_epoch_loss = 0
        iteration_count = 0
        for sent_idx, sentence in enumerate(training_data):
            model.initial_parser_model(sentence['idx_subword'], sentence['idx_word_can'], sentence['idx_bpos_can'],
                                       sentence['only_subword'])

            all_parser_loss = 0
            for gold_action, feasible_action in zip(sentence['gold_actions'], sentence['feasible_actions']):
                train_loss = model.train(gold_action, feasible_action)
                all_parser_loss += train_loss
                model.take_action(gold_action)

            parser_loss = all_parser_loss / len(sentence['gold_actions'])
            all_epoch_loss += all_parser_loss
            iteration_count += len(sentence['gold_actions'])
            if sent_idx % 5 == 0:
                print('parser', sent_idx, ', loss = ', parser_loss)

        epoch_loss = all_epoch_loss / iteration_count
        print('epoch loss:', epoch_loss)

        epoch_count += 1

        break

