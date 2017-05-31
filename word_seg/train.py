from word_seg.data_prep import get_input_batch, read_transition_prob
from word_seg.segment_model import SegmentModel
from word_seg.evaluate import get_head_end_list, evaluate_sentence, evaluate_epoch
from tools.embedding_reader import NetworkParams


def write_log(log_directory, epoch_dict, epoch_count):
    log_file_path = log_directory + 'epoch' + str(epoch_count)
    with open(log_file_path, 'w') as log_file:
        log_file.write(str(epoch_dict) + '\n')


def train(options):
    # prepare data
    network_params = NetworkParams()
    network_params.set_from_options(options, ['subword', 'subword_bigram', 'ws_label'])
    training_feeder, _ = get_input_batch(options['train_file_path'], options['train_subword_file_path'],
                                      network_params, options)
    eval_feeder, _ = get_input_batch(options['eval_file_path'], options['eval_subword_file_path'],
                                  network_params, options)
    transition_prob = read_transition_prob(options['transition_prob_file'])

    parameters = {
        'subword_embedding': network_params.params['subword_embedding'],
        'bigram_embedding': network_params.params['bigram_embedding']
    }

    # for evaluation
    eval_label_list = list()
    while not eval_feeder.is_epoch_end():
        _, _, labels = eval_feeder.get_next_batch()
        head_end_list, word_count = get_head_end_list(labels)
        eval_label_list.append((labels, head_end_list, word_count))

    # start training
    model = SegmentModel(transition_prob, parameters=parameters)
    # model = SegmentModel(transition_prob, model_path=options['ws_model_load_path'])
    epoch_count = 0
    score_list = list()
    max_score = 0

    while True:
        print('epoch', epoch_count)

        training_feeder.shuffle()
        iteration_count = 0
        loss_sum = 0
        while not training_feeder.is_epoch_end():
            subwords, bigrams, labels = training_feeder.get_next_batch()
            sent_loss = model.train(subwords, bigrams, labels)
            loss_sum += sent_loss
            iteration_count += 1
            if iteration_count % 100 == 0:
                print('iteration: ', iteration_count, ', loss = ', sent_loss)

        epoch_loss = loss_sum / iteration_count
        print('epoch loss', epoch_loss)

        # evaluate
        eval_feeder.reset()
        eval_sent_count = 0
        evaluation_list = list()
        while not eval_feeder.is_epoch_end():
            subwords, bigrams, _ = eval_feeder.get_next_batch()
            possible_outputs = model.predict(subwords, bigrams, 1)
            (gold_labels, gold_he_list, gold_word_count) = eval_label_list[eval_sent_count]

            sentence_eval = list()
            for candidate_outputs in possible_outputs:
                eval_dict = evaluate_sentence(candidate_outputs, gold_labels, gold_he_list, gold_word_count)
                sentence_eval.append(eval_dict)
            evaluation_list.append(sentence_eval)

            eval_sent_count += 1

        epoch_eval = evaluate_epoch(evaluation_list)[0]
        tagged_eval = evaluate_epoch(evaluation_list)[1]
        write_log(options['ws_log_dir'], epoch_eval, epoch_count)

        f1_score = epoch_eval['f1_score']
        tag_acc = epoch_eval['tag_accuracy']
        print('tagging accuracy', tag_acc)
        print('f1 score', f1_score)
        print('debug: ', tagged_eval['tag_accuracy'], tagged_eval['f1_score'])
        print('---------------------------------------')
        score_list.append(f1_score)
        average_score = sum(score_list) / len(score_list)

        if f1_score > max_score:
            max_score = f1_score
            model.save_model(options['ws_model_save_path'], epoch_count)

        if f1_score < average_score and epoch_count > 20:
            break

        epoch_count += 1
