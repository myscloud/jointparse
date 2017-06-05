import numpy as np


def calculate_accuracy_one_tag(gold_tags, predicted_tags):
    all_count = len(gold_tags)
    correct_count = 0
    for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
        max_index = np.argmax(predicted_tag)
        print(max_index, gold_tag)
        if max_index == gold_tag:
            correct_count += 1

    return ((correct_count / all_count) * 100), correct_count, all_count
