import tools.data_transform as data_trans

from pos_tag.evaluate import evaluate_accuracy


def predict(model, lang_params, pos_data, data_feeder, data_set_name, candidate_no, pos_type):
    predicted_results = list()
    data_feeder.reset()
    correct_count_sum = tag_count_sum = 0

    while not data_feeder.is_epoch_end():
        x, y, sent_len, token_map = data_feeder.get_next_batch()
        predicted_y = model.predict(x, sent_len)
        predicted_label = data_trans.argmax_max_k(predicted_y, candidate_no)
        predicted_results.extend(predicted_label.tolist())

        _, correct_count, tag_count = evaluate_accuracy(y, predicted_y, token_map)
        correct_count_sum += correct_count
        tag_count_sum += tag_count

    data_accuracy = (correct_count_sum / tag_count_sum) * 100
    print(data_set_name, 'accuracy: ', data_accuracy)

    map_name = {'bpos': 'reverse_bpos_map', 'pos': 'reverse_pos_map'}
    pos_data.map_results_back(predicted_results, lang_params.params[map_name[pos_type]])
    return data_accuracy

