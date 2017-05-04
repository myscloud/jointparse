from collections import deque


class ParserConfig:
    def __init__(self, command, params):
        if command == 'initial':
            self.buffer = deque(['subword' + str(i) for i in range(1, params['subword_idx']+1)])
            self.stack = deque(['word0'])
        else:
            self.buffer = params['buffer']
            self.stack = params['stack']
            if command == 'remove s1':
                self.stack.pop()
            elif command == 'remove s2':
                tmp = self.stack.pop()
                self.stack.pop()
                self.stack.append(tmp)
            elif command == 'shift':
                self.buffer.popleft()
                self.stack.append(params['new_stack_index'])
            elif command == 'remove b1':
                self.buffer.popleft()


class Parser:
    def __init__(self, input_sentence, network_params, labels=None):
        """
        :param input_sentence: A list in from each sentence in tools.ParserInputData
        :param network_params: NetworkParams from tools.embedding_reader
        :param labels: [data_label, subword_label] from each sentence in tools.ExperimentData
        """
        self.data = input_sentence
        self.network_params = network_params

        self.features = dict()
        self.max_stack_index = 0
        self.data_map = dict()
        self.results = list()
        self.arcs = list()
        self.actions = list()

        self.current_config = self.read_initial_settings(input_sentence)

        if labels:
            [self.data_label, self.subword_label] = labels
            self.max_dependent_idx = self.get_max_dependent_node_idx()
        else:
            self.data_label, self.subword_label = None, None

    def read_initial_settings(self, input_sentence):
        initial_config = ParserConfig('initial', {'subword_idx': len(input_sentence)})

        self.features['word0'] = self.get_new_stack_node_features('<S>', 'X')
        self.results.append({'word': '<S>', 'pos': 'X', 'head_idx': None, 'dep_label': None})
        self.data_map['word0'] = 0

        for input_idx, input_info in enumerate(input_sentence):
            label = 'subword' + str(input_idx + 1)
            self.data_map[label] = input_idx
            self.features[label] = self.get_new_buffer_node_features(
                input_info.subword, input_info.word_candidates, input_info.pos_candidates)

        return initial_config

    def is_terminated(self):
        return (len(self.current_config.stack) == 1 and
                len(self.current_config.buffer) == 0 and
                len(self.arcs) == len(self.results) - 1)

    # ---------------------------------------- Features ----------------------------------------

    def get_features(self, feature_pattern):
        features = dict()
        node_dict = self.get_node_dict(feature_pattern.node_list, feature_pattern.relative_node_list)
        for feature_category in feature_pattern.feature_dict:
            features[feature_category] = list()
            for (node_name, param) in feature_pattern.feature_dict[feature_category]:
                node_id = node_dict[node_name]
                if node_id and self.features[node_id]:
                    value = self.features[node_id][param]
                else:
                    value = feature_pattern.none_index[feature_category]
                features[feature_category].append(value)

        return features

    def get_node_dict(self, node_list, relative_node_list):
        node_dict = dict()
        stack = self.current_config.stack
        buffer = self.current_config.buffer

        for node_name in node_list:
            node_index = int(node_name[1:])
            if node_name[0] == 'b' and len(buffer) >= node_index:
                node_id = buffer[node_index-1]
            elif node_name[0] == 's' and len(stack) >= node_index:
                node_id = stack[-node_index]
            else:
                node_id = None
            node_dict[node_name] = node_id

        for node_name in relative_node_list:
            parent_node, relation = node_name.split(':')
            if node_dict[parent_node]:
                parent_node_id = node_dict[parent_node]
                node_dict[node_name] = self.features[parent_node_id][relation]
            else:
                node_dict[node_name] = None

        return node_dict

    def get_new_buffer_node_features(self, subword, candidate_words, candidate_bpos):
        new_features = dict()
        unk_index = self.network_params.params['subword_map']['<UNK>']
        new_features['s'] = self.network_params.params['subword_map'].get(subword, unk_index)
        words_index = self.network_params.map_list_with_params(candidate_words, 'word_map', '<UNK>')
        bpos_index = self.network_params.map_list_with_params(candidate_bpos, 'bpos_map')
        for word_idx, cword_idx in enumerate(words_index):
            label = 'w' + str(word_idx+1)  # w1, w2, w3
            new_features[label] = cword_idx

        for bpos_idx, cbpos_idx in enumerate(bpos_index):
            label = 'bp' + str(bpos_idx+1)  # bp1, bp2, bp3
            new_features[label] = cbpos_idx

        return new_features

    def get_new_stack_node_features(self, word, pos):
        new_features = {'l1': None, 'l2': None, 'll1': None, 'r1': None, 'r2': None, 'rr1': None}
        unk_index = self.network_params.params['word_map']['<UNK>']
        new_features['w'] = self.network_params.params['word_map'].get(word, unk_index)
        new_features['p'] = self.network_params.params['pos_map'].get(pos)
        new_features['l'] = None

        return new_features

    # ---------------------------------------- Parser Action ----------------------------------------

    def add_new_arc(self, head_node_id, dep_label,  dependent_node_id, arc_type):
        if arc_type == 'left':
            child1, child2, child_of_child = 'l1', 'l2', 'll1'
        else:
            child1, child2, child_of_child = 'r1', 'r2', 'rr1'

        self.features[head_node_id][child2] = self.features[head_node_id][child1]
        self.features[head_node_id][child1] = dependent_node_id
        if self.features[dependent_node_id][child1]:  # child of child
            self.features[head_node_id][child_of_child] = self.features[dependent_node_id][child1]

        self.features[dependent_node_id]['l'] = self.network_params.params['dep_label_map'][dep_label]
        self.arcs.append((head_node_id, dep_label, dependent_node_id))

        head_node_index = self.data_map[head_node_id]
        dependent_node_index = self.data_map[dependent_node_id]
        self.results[dependent_node_index]['head_idx'] = head_node_index
        self.results[dependent_node_index]['dep_label'] = dep_label

    def add_new_stack_node(self, buffer_label, pos):
        buffer_data_id = self.data_map[buffer_label]
        word = self.data[buffer_data_id].subword
        subword_idx = self.data[buffer_data_id].subword_idx
        self.max_stack_index += 1
        label = 'word' + str(self.max_stack_index)

        result_dict = {'word': word, 'pos': pos, 'head_idx': None, 'dep_label': None}
        self.results.append(result_dict)
        self.data_map[label] = len(self.results) - 1
        self.features[label] = self.get_new_stack_node_features(word, pos)

        return label

    def append_to_top_of_stack(self, tos_label, buffer_label, pos):
        buffer_data_id = self.data_map[buffer_label]
        appending_subword = self.data[buffer_data_id].subword
        appending_subword_idx = self.data[buffer_data_id].subword_idx

        tos_result_index = self.data_map[tos_label]
        new_word = self.results[tos_result_index]['word'] + appending_subword

        # edit results
        self.results[tos_result_index]['word'] = new_word
        self.results[tos_result_index]['pos'] = pos

        # edit features
        unk_index = self.network_params.params['word_map']['<UNK>']
        self.features[tos_label]['w'] = self.network_params.params['word_map'].get(new_word, unk_index)
        self.features[tos_label]['p'] = self.network_params.params['pos_map'].get(pos)

    def get_max_dependent_node_idx(self):
        max_idx = [-1] * len(self.data_label)
        for word_idx, word_info in enumerate(self.data_label):
            head_idx = word_info.head_idx
            max_idx[head_idx-1] = word_idx

        return max_idx

    def get_next_gold_standard_action(self):
        if not (self.data_label and self.subword_label):
            raise Exception('No gold labels defined for the parser.')

        # consider s1, s2, and b1
        config = self.current_config

        stack2_data_idx = self.data_map[config.stack[-2]] if len(config.stack) >= 2 else None
        stack2_gold_data = self.data_label[stack2_data_idx-1] if stack2_data_idx else None  # if not none and != 0

        stack1_data_idx = self.data_map[config.stack[-1]] if len(config.stack) >= 1 else None
        stack1_gold_data = self.data_label[stack1_data_idx-1] if stack1_data_idx else None  # if not none and != 0

        buffer_data_idx = self.data_map[config.buffer[0]] if len(config.buffer) >= 1 else None
        buffer_word_idx = self.subword_label[buffer_data_idx].word_idx if type(buffer_data_idx) is int else None

        if type(buffer_word_idx) is int and buffer_word_idx == stack1_data_idx:  # buffer1 and stack1 is in same word
            pos_tag = self.data_label[buffer_word_idx - 1].pos
            action = ('APPEND', pos_tag)
        elif (stack2_gold_data is not None) and stack2_gold_data.head_idx == stack1_data_idx:
            dep_label = stack2_gold_data.dep_label
            action = ('LEFT-ARC', dep_label)
        elif (stack2_data_idx is not None) and stack1_gold_data.head_idx == stack2_data_idx \
                and self.max_stack_index > self.max_dependent_idx[stack1_data_idx-1]:
            # if head_idx of stack2 is stack1 and all stack1's dependents are already popped from the stack
            dep_label = stack1_gold_data.dep_label
            action = ('RIGHT-ARC', dep_label)
        else:
            pos_tag = self.data_label[buffer_word_idx-1].pos
            action = ('SHIFT', pos_tag)

        return action

    def is_similar_to_gold_label(self):
        if len(self.results) - 1 != len(self.data_label):
            return False

        for tagged_info, gold_info in zip(self.results[1:], self.data_label):
            for attribute in tagged_info:
                if tagged_info[attribute] != getattr(gold_info, attribute):
                    return False

        return True
