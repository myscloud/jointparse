from lstm_parser.parser import Parser
from tools.experiment_data import ExperimentData


def prepare_parser_data(options, params, data_types):
    # read data
    data_file_path = options[data_types + '_file_path']
    subword_file_path = options[data_types + '_subword_file_path']
    word_candidates_path = options[data_types + '_word_seg_path']
    bpos_candidates_path = options[data_types + '_pos_candidates_path']

    k_words = options['no_word_candidate']

    exp_data = ExperimentData(data_file_path, subword_file_path)
    word_candidates = read_word_candidates(word_candidates_path, k_words)
    mapped_word_candidates = generate_subwords_word_candidate(exp_data.subword, word_candidates)
    bpos_candidates = read_bpos_candidates(bpos_candidates_path)

    # for each sentence, prepare data for parser
    parser_data = list()
    idx = 0
    for gold_data, subword, word_can, bpos_can in zip(exp_data.data,
                                                      exp_data.subword, mapped_word_candidates, bpos_candidates):
        sentence_subword = list()
        for subword_idx in range(len(subword)):
            curr_subword = subword[subword_idx].subword
            word_info = dict()
            word_info['subword'] = curr_subword
            word_info['word_candidates'] = word_can[subword_idx]
            word_info['bpos_candidates'] = bpos_can[subword_idx]
            sentence_subword.append(word_info)
        # generate gold action list
        action_list, feasible_action_list, action_idx = get_gold_action_list(sentence_subword, gold_data, subword)
        feasible_action_index = [get_feasible_action_index(actions, params.params['reverse_action_map'])
                                 for actions in feasible_action_list]

        sentence_info = dict()
        sentence_info['sentence_data'] = sentence_subword
        sentence_info['gold_data'] = gold_data
        sentence_info['gold_actions'] = params.map_list_with_params(action_list, 'action_map')
        sentence_info['pseudo_label'] = get_pseudo_label(action_idx)
        sentence_info['feasible_actions'] = feasible_action_index
        sentence_info['buffer_packet'] = get_buffer_packet(gold_data, subword)

        subword_list = [subword_info.subword for subword_info in subword]
        sentence_info['only_subword'] = subword_list
        sentence_info['idx_subword'] = params.map_list_with_params(subword_list, 'subword_map', '<UNK>')
        sentence_info['idx_word_can'] = [params.map_list_with_params(sub_word_can, 'word_map', '<UNK>') for sub_word_can in word_can]
        sentence_info['idx_bpos_can'] = [params.map_list_with_params(sub_bpos_can, 'bpos_map') for sub_bpos_can in bpos_can]
        sentence_info['idx_buffer_packet'] = get_mapped_buffer_packet(sentence_info['buffer_packet'], params)
        parser_data.append(sentence_info)

    return parser_data


def get_pseudo_label(action_idx):
    new_action = action_idx + [4, 4]
    pseudo_labels = list()

    for i in range(0, len(action_idx)):
        new_label = (5 * new_action[i+1]) + new_action[i+2]
        pseudo_labels.append(new_label)

    return pseudo_labels


def get_buffer_packet(sent_data, subword_data):
    all_words = list()
    curr_subword = list()
    curr_word_idx = 1

    for subword_idx in range(0, len(subword_data) + 1):
        subword_info = subword_data[subword_idx] if subword_idx < len(subword_data) else None
        if subword_info is None or subword_info.word_idx != curr_word_idx:
            word_info = {
                'word': sent_data[curr_word_idx - 1].word,
                'pos': sent_data[curr_word_idx - 1].pos,
                'subword': curr_subword.copy()
            }
            all_words.append(word_info)
            curr_subword = list()

        if subword_info is None:
            break

        curr_word_idx = subword_info.word_idx
        curr_subword.append(subword_info.subword)

    return all_words


def get_mapped_buffer_packet(buffer_packet, params):
    mapped_list = list()
    for word_info in buffer_packet:
        mapped_info = {
            'word': params.params['word_map'].get(word_info['word'], 0),
            'pos': params.params['pos_map'].get(word_info['pos']),
            'subword': params.map_list_with_params(word_info['subword'], 'subword_map', '<UNK>')
        }
        mapped_list.append(mapped_info)

    return mapped_list


def read_word_candidates(word_candidates_path, k_words):
    candidates = list()
    curr_sentence = list()

    with open(word_candidates_path) as candidate_file:
        for line in candidate_file:
            curr_sentence.append(line.strip().split(' '))

            if len(curr_sentence) == k_words:
                candidates.append(curr_sentence)
                curr_sentence = list()

    return candidates


def generate_subwords_word_candidate(subword_list, word_candidate_list):
    word_candidate = list()
    for subword_sent, candidate_sent in zip(subword_list, word_candidate_list):
        subword_word_list = [list() for _ in range(len(subword_sent))]

        for candidate in candidate_sent:
            curr_word_idx = 0
            curr_subword_idx = 0

            for subword_idx, subword in enumerate([subword_info.subword for subword_info in subword_sent]):
                if subword not in candidate[curr_word_idx]:
                    raise Exception('Subword and word candidate don\'t match')
                subword_word_list[subword_idx].append(candidate[curr_word_idx])
                curr_subword_idx += len(subword)
                if curr_subword_idx == len(candidate[curr_word_idx]):
                    curr_word_idx += 1
                    curr_subword_idx = 0

        word_candidate.append(subword_word_list)

    return word_candidate


def read_bpos_candidates(bpos_candidates_path):
    candidates = list()
    curr_sentence = list()

    with open(bpos_candidates_path) as candidate_file:
        for line in candidate_file:
            if len(line) > 1 and line[0] != '#':
                tokens = line.strip().split('\t')
                pos_candidate = tokens[2].split(',')
                curr_sentence.append(pos_candidate)
            elif line[0] == '#' and len(curr_sentence) > 0:
                candidates.append(curr_sentence)
                curr_sentence = list()

        if len(curr_sentence) > 0:
            candidates.append(curr_sentence)

    return candidates


action_name_list = ['LEFT-ARC', 'RIGHT-ARC', 'SHIFT', 'APPEND']


def get_gold_action_list(sentence_data, gold_data, subword_data):
    parser = Parser(sentence_data)

    max_dependent_idx = [-1] * (len(gold_data) + 1)
    for word_idx, word_info in enumerate(gold_data):
        head_idx = word_info.head_idx
        max_dependent_idx[head_idx] = word_idx + 1

    action_list = list()
    action_idx_list = list()
    feasible_action_list = list()

    while not parser.is_parsing_terminated():
        stack2, stack1, buffer = parser.get_current_configuration()

        action, params = None, None
        if buffer is not None and subword_data[buffer - 1].word_idx == stack1:
            action = 'APPEND'
            word_idx = subword_data[buffer - 1].word_idx
            params = gold_data[word_idx - 1].pos
        elif stack1 is not None and stack2 is not None and stack2 > 0 and gold_data[stack2 - 1].head_idx == stack1:
            action = 'LEFT-ARC'
            params = gold_data[stack2 - 1].dep_label
        elif stack1 is not None and stack2 is not None and gold_data[stack1 - 1].head_idx == stack2\
                and len(parser.results) > max_dependent_idx[stack1]:
            action = 'RIGHT-ARC'
            params = gold_data[stack1 - 1].dep_label
        elif buffer is not None:
            action = 'SHIFT'
            word_idx = subword_data[buffer - 1].word_idx
            params = gold_data[word_idx - 1].pos

        action_tuple = (action, params)
        action_list.append(action_tuple)
        action_idx_list.append(action_name_list.index(action))

        feasible_actions = parser.get_feasible_actions()
        feasible_action_list.append(feasible_actions)

        parser.take_action(action_tuple)

    return action_list, feasible_action_list, action_idx_list


def get_word_label(gold_subword_data):
    tag_count = 0
    tag_list = [0]

    for i in range(0, len(gold_subword_data)+1):
        tag_count = (tag_count % 16) * 2
        if (i < len(gold_subword_data) - 1) and (gold_subword_data[i].word_idx == gold_subword_data[i+1].word_idx):
            tag_count += 1
        tag_list.append(tag_count)

    return tag_list[2:]


def get_feasible_action_index(feasible_actions, reverse_map):
    index_list = list()
    for action in action_name_list:
        if action in feasible_actions:
            index_list.append(1)
        else:
            index_list.append(0)
    return index_list

#
# def get_feasible_action_index(feasible_actions, reverse_map):
#     index_list = list()
#     for i in range(len(reverse_map)-1):
#         if reverse_map[i][0] in feasible_actions:
#             index_list.append(1)
#         else:
#             index_list.append(0)
#     return index_list
