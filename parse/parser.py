from collections import deque


class ParserConfig:
    def __init__(self, command, params):
        if command == 'initial':
            self.buffer = deque(['subword' + str(i) for i in range(1, params+1)])
            self.stack = deque(['word0'])
        else:
            self.buffer = params.buffer
            self.stack = params.stack
            if command == 'remove s1':
                self.stack.pop()
            elif command == 'remove s2':
                tmp = self.stack.pop()
                self.stack.pop()
                self.stack.append(tmp)
            elif command == 'shift':
                tmp = self.buffer.popleft()
                self.stack.append(tmp)
            elif command == 'remove b1':
                self.buffer.popleft()


class Parser:
    def __init__(self, input_sentence, network_params, label=None):
        self.data = input_sentence
        self.network_params = network_params
        self.features = dict()
        self.max_stack_index = 0
        self.current_config = self.read_initial_settings(input_sentence)

        self.arcs = list()

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

    def read_initial_settings(self, input_sentence):
        initial_config = ParserConfig('initial', len(input_sentence))
        self.features['word0'] = self.get_new_stack_node_features('<S>', 'X')
        for input_idx, input_info in enumerate(input_sentence):
            label = 'subword' + str(input_idx + 1)
            self.features[label] = self.get_new_buffer_node_features(
                input_info.subword, input_info.word_candidates, input_info.pos_candidates)

        return initial_config
