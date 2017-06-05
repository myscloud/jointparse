from tools.experiment_data import ExperimentData
from parse.evaluate import get_parser_evaluation, get_epoch_evaluation


def evaluate_baseline(options):
    predicted_data = ExperimentData(options['baseline_test_file'], options['baseline_test_subword_file'])
    gold_data = ExperimentData(options['test_file_path'], options['test_subword_file_path'])

    evaluation_list = list()
    sent_count = 1
    for predicted_sent, gold_sent in zip(predicted_data.data, gold_data.data):
        print(sent_count)
        reformatted_sent = list()
        for word_info in predicted_sent:
            new_word = dict()
            new_word['word'] = word_info.word
            new_word['pos'] = word_info.pos
            new_word['head_idx'] = word_info.head_idx
            new_word['dep_label'] = word_info.dep_label
            reformatted_sent.append(new_word)

        eval_dict = get_parser_evaluation(reformatted_sent, gold_sent)
        evaluation_list.append(eval_dict)
        sent_count += 1

    all_eval = get_epoch_evaluation(evaluation_list)
    print(all_eval)
