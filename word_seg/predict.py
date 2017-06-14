from tools.embedding_reader import NetworkParams
from word_seg.data_prep import get_input_batch, read_transition_prob, write_results
from word_seg.segment_model import SegmentModel
from word_seg.evaluate import evaluate_sentence, evaluate_epoch, get_head_end_list


def predict(options):
    # data_file_type = ['test']
    data_file_type = ['training', 'eval', 'dev', 'test']
    data_file_dict = {
        'training': (options['train_file_path'], options['train_subword_file_path']),
        'eval': (options['eval_file_path'], options['eval_subword_file_path']),
        'dev': (options['dev_file_path'], options['dev_subword_file_path']),
        'test': (options['test_file_path'], options['test_subword_file_path']),
    }
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'subword_bigram', 'ws_label'])

    transition_prob = read_transition_prob(options['transition_prob_file'])
    model = SegmentModel(transition_prob, model_path=options['ws_model_load_path'])

    for file_type in data_file_type:
        print('writing ', file_type, 'data')
        (file_path, subword_file_path) = data_file_dict[file_type]
        data_feeder, exp_data = get_input_batch(file_path, subword_file_path, network_params, options)
        evaluation_list = list()
        subword_list = list()
        output_list = list()

        for sentence in exp_data.subword:
            subword_list.append([subword_info.subword for subword_info in sentence])

        while not data_feeder.is_epoch_end():
            subwords, bigrams, labels, _ = data_feeder.get_next_batch()
            possible_outputs = model.predict(subwords, bigrams, 3)

            gold_head_end_list, gold_count = get_head_end_list(labels)
            sentence_eval = list()
            for candidate_output in possible_outputs:
                eval_dict = evaluate_sentence(candidate_output, labels, gold_head_end_list, gold_count)
                sentence_eval.append(eval_dict)

            evaluation_list.append(sentence_eval)
            output_list.append(possible_outputs)

        all_eval = evaluate_epoch(evaluation_list)
        eval_file_path = options['ws_results_path'] + file_type + '.eval'
        with open(eval_file_path, 'w') as eval_file:
            eval_file.write(str(all_eval) + '\n')

        output_file_path = options['ws_results_path'] + file_type + '.ws'
        write_results(output_file_path, subword_list, output_list)
