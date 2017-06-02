from tools.embedding_reader import NetworkParams
from multilang_pos.data_prep import prepare_input_data, get_input_batch, read_word_freq_map
from multilang_pos.tagger_model import TaggerModel
from multilang_pos.evaluate import calculate_accuracy_one_tag
import multilang_pos.data_prep as pos_data


def train(options):
    # prepare data
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'word', 'pos'])

    log_freq_map = read_word_freq_map(options['log_freq_file'])
    data_set_files = dict()
    data_set_files['training'] = (options['train_file_path'], options['train_subword_file_path'])
    data_set_files['eval'] = (options['eval_file_path'], options['eval_subword_file_path'])
    data_set = dict()

    for data_set_name in data_set_files:
        data_file_path, subword_file_path = data_set_files[data_set_name]
        data_set[data_set_name] = prepare_input_data(data_file_path, subword_file_path,
                     network_params.params['word_map'], network_params.params['subword_map'],
                     network_params.params['pos_map'], log_freq_map, pos_data.use_defined_subword)
    training_feeder = get_input_batch(data_set['training'], 1)
    eval_feeder = get_input_batch(data_set['eval'], 1)

    embeddings = [network_params.params['word_embedding'], network_params.params['subword_embedding']]
    model = TaggerModel(embeddings=embeddings)
    epoch_count = 0
    max_eval_acc = 0
    accuracy_list = list()

    # generate eval gold tags
    raw_gold_tags = data_set['eval']['pos_labels']
    eval_gold_tags = [tag for sentence in raw_gold_tags for tag in sentence]

    # start training!
    while True:
        print('epoch', epoch_count)

        # training
        training_feeder.shuffle()
        iteration_count = 0
        loss_sum = 0
        while not training_feeder.is_epoch_end():
            batch_input = training_feeder.get_next_batch()
            input_list = batch_input[0:3]
            label_list = batch_input[-2:]
            iter_loss = model.train(input_list, label_list)

            loss_sum += iter_loss
            iteration_count += 1
            if iteration_count % 100 == 0:
                print('iteration: ', iteration_count, ', loss: ', iter_loss)

        epoch_loss = loss_sum / iteration_count
        print('epoch loss: ', epoch_loss)

        eval_feeder.reset()
        all_predicted_tags = list()
        while not eval_feeder.is_epoch_end():
            batch_input = eval_feeder.get_next_batch()
            input_list = batch_input[0:3]
            predicted_tags = model.predict(input_list)
            all_predicted_tags.extend(predicted_tags)

        accuracy, correct_count, all_count = calculate_accuracy_one_tag(eval_gold_tags, all_predicted_tags)
        accuracy_list.append(accuracy)
        print('eval accuracy:', accuracy, '(', correct_count, 'from', all_count, ')')

        if accuracy > max_eval_acc:
            model.save_model(options['pos_model_save_path'], epoch_count)
            max_eval_acc = accuracy

        print('----------------------------')
        epoch_count += 1
        average_acc = sum(accuracy_list) / len(accuracy_list)
        if accuracy < average_acc:
            break

