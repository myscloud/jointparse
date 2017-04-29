from math import ceil
from collections import deque

from blaze.compute.numpy import epoch

from tools.experiment_data import ExperimentData
from tools.embedding_reader import NetworkParams
from tools.batch_reader import BatchReader
import tools.data_transform as data_trans

from pos_tag.data_pos import DataPOS
from pos_tag.tagger_model import TaggerModel
from pos_tag.evaluate import evaluate_accuracy
from pos_tag.predict import predict


def get_data_list_for_batch_reader(data, pad_element):
    data_list = [
        {'data': data.phrases, 'pad_element': pad_element, 'post_func': data_trans.to_matrix},
        {'data': data.phrases_label, 'pad_element': pad_element, 'post_func': data_trans.to_matrix},
        {'data': data.phrase_len, 'pad_element': 1, 'post_func': None},
        {'data': data.phrase_token_map, 'pad_element': (0, 0), 'post_func': None}
    ]
    return data_list


def train(options):
    lang_params = NetworkParams()
    lang_params.set_subword_embedding(options['subword_embedding'], options['subword_embedding_map'])
    lang_params.set_index_file('bpos_map', 'reverse_bpos_map', options['bpos_map'])

    raw_train_data = ExperimentData(options['train_file_path'], options['train_subword_file_path'])
    raw_eval_data = ExperimentData(options['eval_file_path'], options['eval_subword_file_path'])
    raw_dev_data = ExperimentData(options['dev_file_path'], options['dev_subword_file_path'])
    raw_test_data = ExperimentData(options['test_file_path'], options['test_subword_file_path'])

    n_len = options['training_sentence_len']
    overlap = options['sentence_overlapping_chars']
    train_data = DataPOS(raw_train_data.subword, lang_params, n_len, overlap, data_prop=raw_train_data.data)
    eval_data = DataPOS(raw_eval_data.subword, lang_params, n_len, overlap, data_prop=raw_eval_data.data)
    dev_data = DataPOS(raw_dev_data.subword, lang_params, n_len, overlap, data_prop=raw_dev_data.data)
    test_data = DataPOS(raw_test_data.subword, lang_params, n_len, overlap, data_prop=raw_test_data.data)

    train_data_list = [
        {'data': train_data.phrases, 'pad_element': None, 'post_func': data_trans.to_matrix},
        {'data': train_data.phrases_label, 'pad_element': None, 'post_func': data_trans.to_matrix},
        {'data': train_data.phrase_len, 'pad_element': None, 'post_func': None}
    ]

    pad_element = [0] * n_len
    eval_train_data_list = get_data_list_for_batch_reader(train_data, pad_element)
    eval_data_list = get_data_list_for_batch_reader(eval_data, pad_element)
    dev_data_list = get_data_list_for_batch_reader(dev_data, pad_element)
    test_data_list = get_data_list_for_batch_reader(test_data, pad_element)

    train_data_feeder = BatchReader(train_data_list, options['pos_batch_size'])
    eval_train_data_feeder = BatchReader(eval_train_data_list, options['pos_batch_size'])
    eval_data_feeder = BatchReader(eval_data_list, options['pos_batch_size'])
    dev_data_feeder = BatchReader(dev_data_list, options['pos_batch_size'])
    test_data_feeder = BatchReader(test_data_list, options['pos_batch_size'])

    train_batch_count = ceil(len(train_data.phrases) / options['pos_batch_size'])
    eval_batch_count = ceil(len(eval_data.phrases) / options['pos_batch_size'])

    epoch_count = 0
    latest_accuracies = deque(maxlen=10)
    model = TaggerModel(lang_params.params['subword_embedding'])
    while True:
        print('epoch ', epoch_count)
        loss_sum = 0
        train_data_feeder.shuffle()
        while not train_data_feeder.is_epoch_end():
            x, y, sent_len = train_data_feeder.get_next_batch()
            batch_loss = model.train(x, y, sent_len)
            loss_sum += batch_loss
        train_loss = loss_sum / train_batch_count

        loss_sum = 0
        correct_count_sum = tag_count_sum = 0
        eval_data_feeder.shuffle()
        results = list()
        while not eval_data_feeder.is_epoch_end():
            x, y, sent_len, token_map = eval_data_feeder.get_next_batch()
            predicted_y, batch_loss = model.evaluate(x, y, sent_len)
            results.append(predicted_y)
            loss_sum += batch_loss

            _, correct_count, tag_count = evaluate_accuracy(y, predicted_y, token_map)
            correct_count_sum += correct_count
            tag_count_sum += tag_count

        eval_accuracy = (correct_count_sum / tag_count_sum) * 100
        latest_accuracies.append(eval_accuracy)
        eval_loss = loss_sum / eval_batch_count

        print('train loss', train_loss)
        print('eval loss', eval_loss)
        print('eval accuracy', eval_accuracy)

        average_accuracy = sum(latest_accuracies) / len(latest_accuracies)
        if eval_accuracy < average_accuracy:
            train_data.write_tagged_results(options['pos_results_path'] + 'train.pos')
            eval_data.write_tagged_results(options['pos_results_path'] + 'eval.pos')
            dev_data.write_tagged_results(options['pos_results_path'] + 'dev.pos')
            test_data.write_tagged_results(options['pos_results_path'] + 'test.pos')
            break

        model.save_model(options['pos_model_save_path'], epoch_count)
        epoch_count += 1

        # get prediction results
        max_k = options['no_pos_candidates']
        predict(model, lang_params, train_data, eval_train_data_feeder, 'train', max_k)
        predict(model, lang_params, eval_data, eval_data_feeder, 'eval', max_k)
        predict(model, lang_params, dev_data, dev_data_feeder, 'dev', max_k)
        predict(model, lang_params, test_data, test_data_feeder, 'test', max_k)
