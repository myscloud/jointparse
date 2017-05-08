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

    sum_f1 = [0] * options['no_word_candidate']
    for sentence, gold_sentence in zip(ws_candidates, gold_data.data):
        for can_idx, candidate in enumerate(sentence):
            _, correct_count = map_predicted_with_gold_words(candidate, gold_sentence)
            precision = (correct_count / len(sentence)) * 100
            recall = (correct_count / len(gold_sentence)) * 100
            f1_score = get_f1_score(precision, recall)
            sum_f1[can_idx] += f1_score

    print('all sentence: ', len(ws_candidates))
    for can_idx in range(len(sum_f1)):
        print(can_idx, ':', sum_f1[can_idx]/len(ws_candidates))
