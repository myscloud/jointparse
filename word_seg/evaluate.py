
def get_head_end_list(labels_list):
    word_count = 0
    head_end_list = list()
    last_start = 0

    for label_idx, label in enumerate(labels_list):
        head_end_list.append(None)
        if label == 0 or label == 3:
            last_start = label_idx

        head_end_list[last_start] = label_idx

        if label == 2 or label == 3:
            last_start = label_idx + 1

    for end_idx in head_end_list:
        if end_idx is not None:
            word_count += 1

    return head_end_list, word_count


def get_tag_accuracy(gold_labels, predicted_labels):
    correct_count = 0
    for gold, predict in zip(gold_labels, predicted_labels):
        if gold == predict:
            correct_count += 1

    return correct_count, (correct_count / len(gold_labels)) * 100


def get_correct_word_count(gold_he, predicted_he):
    correct_count = 0
    for gold_end, predicted_end in zip(gold_he, predicted_he):
        if gold_end is not None and predicted_end is not None and gold_end == predicted_end:
            correct_count += 1
    return correct_count


def get_word_f1(gold_he, gold_count, predicted_he, predicted_count):
    correct_count = get_correct_word_count(gold_he, predicted_he)
    precision = (correct_count / predicted_count) * 100
    recall = (correct_count / gold_count) * 100
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return correct_count, precision, recall, f1_score


def evaluate_sentence(predicted_labels, gold_labels, gold_head_end, gold_word_count):
    predicted_head_end, predicted_word_count = get_head_end_list(predicted_labels)
    correct_tags, tag_accuracy = get_tag_accuracy(gold_labels, predicted_labels)
    correct_words, p, r, f1 = get_word_f1(gold_head_end, gold_word_count, predicted_head_end, predicted_word_count)
    evaluation_dict = {
        'correct_tags': correct_tags,
        'correct_words': correct_words,
        'tag_accuracy': tag_accuracy,
        'f1_score': f1,
        'precision': p,
        'recall': r,
        'gold_count': gold_word_count,
        'predicted_count': predicted_word_count,
        'tag_count': len(gold_labels)
    }
    return evaluation_dict


def evaluate_epoch(evaluation_list):
    n_candidates = len(evaluation_list[0])
    n_sent = len(evaluation_list)

    all_gold_count = 0
    all_tag_count = 0
    correct_tags = [0 for _ in range(n_candidates)]
    correct_words = [0 for _ in range(n_candidates)]
    predicted_counts = [0 for _ in range(n_candidates)]
    f1_scores = [0 for _ in range(n_candidates)]
    precision = [0 for _ in range(n_candidates)]
    recall = [0 for _ in range(n_candidates)]

    for sentence in evaluation_list:
        all_gold_count += sentence[0]['gold_count']
        all_tag_count += sentence[0]['tag_count']

        for can_id, candidate in enumerate(sentence):
            correct_tags[can_id] += candidate['correct_tags']
            correct_words[can_id] += candidate['correct_words']
            predicted_counts[can_id] += candidate['predicted_count']
            precision[can_id] += candidate['precision']
            recall[can_id] += candidate['recall']
            f1_scores[can_id] += candidate['f1_score']

    epoch_eval_list = list()
    for can_id in range(n_candidates):
        eval_dict = {
            'correct_tags': correct_tags[can_id],
            'correct_words': correct_words[can_id],
            'tag_accuracy': (correct_tags[can_id] / all_tag_count) * 100,
            'f1_score': f1_scores[can_id] / n_sent,
            'precision': precision[can_id] / n_sent,
            'recall': recall[can_id] / n_sent,
            'gold_count': all_gold_count,
            'predicted_count': predicted_counts[can_id],
            'tag_count': all_tag_count
        }
        epoch_eval_list.append(eval_dict)

    return epoch_eval_list
