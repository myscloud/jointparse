from tools.experiment_data import ExperimentData
from parse.evaluate import map_predicted_with_gold_words, get_f1_score


def read_ws_candidates(file_path, no_candidates):
    sentences = list()
    candidates = list()

    with open(file_path) as ws_file:
        for line in ws_file:
            words = line.strip().split('  ')
            candidates.append([{'word': word} for word in words])
            if len(candidates) == no_candidates:
                sentences.append(candidates)
                candidates = list()

    return sentences


def evaluate_word_segmentation(options):
    ws_candidates = read_ws_candidates(options['eval_word_seg_path'], options['no_word_candidate'])
    gold_data = ExperimentData(options['eval_file_path'], options['eval_subword_file_path'])

    sum_p = [0] * options['no_word_candidate']
    sum_r = [0] * options['no_word_candidate']
    sum_f1 = [0] * options['no_word_candidate']
    for sentence, gold_sentence in zip(ws_candidates, gold_data.data):
        for can_idx, candidate in enumerate(sentence):
            _, correct_count = map_predicted_with_gold_words(candidate, gold_sentence)
            precision = (correct_count / len(candidate)) * 100
            recall = (correct_count / len(gold_sentence)) * 100
            f1_score = get_f1_score(precision, recall)

            sum_p[can_idx] += precision
            sum_r[can_idx] += recall
            sum_f1[can_idx] += f1_score

    print('all sentence: ', len(ws_candidates))
    for can_idx in range(len(sum_f1)):
        print('precision ', can_idx, ':', sum_p[can_idx] / len(ws_candidates))
        print('recall ', can_idx, ':', sum_r[can_idx] / len(ws_candidates))
        print('f1 score ', can_idx, ':', sum_f1[can_idx]/len(ws_candidates))


def evaluate_word_segmentation_coverage(options):
    ws_candidates = read_ws_candidates(options['eval_word_seg_path'], options['no_word_candidate'])
    gold_data = ExperimentData(options['eval_file_path'], options['eval_subword_file_path'])

    no_matched = list()
    no_all = list()

    for sentence, gold_sentence in zip(ws_candidates, gold_data.data):
        gold_matched = set()
        for can_idx, candidate in enumerate(sentence):
            can_gold_map, _ = map_predicted_with_gold_words(candidate, gold_sentence)
            for word_matched in can_gold_map:
                gold_matched.add(word_matched)

        gold_matched.discard(None)
        no_matched.append(len(gold_matched))
        no_all.append(len(gold_sentence))

    n_covered_words = sum(no_matched)
    n_all = sum(no_all)
    print(n_covered_words, n_all)
    print((n_covered_words / n_all) * 100)
