import numpy as np


def evaluate_accuracy(gold_label, predicted_label, evaluated_tokens):
    token_count = 0
    correct_count = 0
    for gold_sent, predicted_sent, eval_sent in zip(gold_label, predicted_label, evaluated_tokens):
        (start, end) = eval_sent
        for gold_tag, predicted_tag_vec in zip(gold_sent[start:end], predicted_sent[start:end]):
            token_count += 1
            if gold_tag == np.argmax(predicted_tag_vec):
                correct_count += 1

    accuracy = (correct_count / token_count) * 100
    return accuracy, correct_count, token_count


def evaluate_from_file(options):
    evaluated_file_path = options['evaluated_pos_file']

    sentence_acc = list()
    all_correct_count = [0] * options['no_pos_candidates']
    all_tag_count = 0
    correct_count = [0] * options['no_pos_candidates']
    tag_count = 0

    with open(evaluated_file_path) as evaluated_file:
        for line in evaluated_file:
            if len(line) <= 1:
                sentence_acc.append((correct_count, tag_count))
                for i in range(len(all_correct_count)):
                    all_correct_count[i] += correct_count[i]
                all_tag_count += tag_count
                correct_count = [0] * options['no_pos_candidates']
                tag_count = 0
            elif line[0] != '#':
                _, _, predicted_pos, real_tag = line.strip().split('\t')
                all_predicted = predicted_pos.split(',')
                for i in range(len(all_predicted)):
                    if all_predicted[i] == real_tag:
                        correct_count[i] += 1
                tag_count += 1

    print(sentence_acc[0])
    print('correct count', all_correct_count)
    print('all tag count', all_tag_count)



