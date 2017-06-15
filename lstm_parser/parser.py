
class Parser:
    def __init__(self, subword_info):
        self.buffer = list()
        for subword_idx, subword in enumerate(subword_info):
            self.buffer.append((subword_idx + 1, subword['subword']))

        self.stack = list()
        self.actions = list()
        self.results = list()
        self.add_to_stack('ROOT', None)
        self.arcs = list()

    def is_parsing_terminated(self):
        return len(self.stack) == 1 and len(self.buffer) == 0

    def get_current_configuration(self):
        stack2 = self.stack[-2] if len(self.stack) >= 2 else None
        stack1 = self.stack[-1] if len(self.stack) >= 1 else None
        buffer = self.buffer[0][0] if len(self.buffer) > 0 else None
        config = (stack2, stack1, buffer)
        return config

    def get_feasible_actions(self):
        feasible_actions = list()
        latest_action = self.actions[-1][0] if len(self.actions) > 0 else None

        if len(self.buffer) > 0:
            feasible_actions.append('SHIFT')
        if len(self.buffer) > 0 and len(self.stack) > 1 and (latest_action == 'SHIFT' or latest_action == 'APPEND'):
            feasible_actions.append('APPEND')
        if len(self.stack) >= 2:
            feasible_actions.append('RIGHT-ARC')
            if self.stack[-2] != 0:  # if left node is not root node, otherwise, right node can't be head of root node
                feasible_actions.append('LEFT-ARC')

        return feasible_actions

    def take_action(self, action_tuple):
        action, params = action_tuple

        feasible_actions = self.get_feasible_actions()
        if action not in feasible_actions:
            return False

        action_map = {
            'SHIFT': self.take_action_shift,
            'APPEND': self.take_action_append,
            'LEFT-ARC': self.take_action_left_arc,
            'RIGHT-ARC': self.take_action_right_arc
        }
        action_map[action](params)
        self.actions.append((action, params))

        return True

    def take_action_shift(self, pos_tag):
        subword = self.buffer[0][1]
        self.buffer = self.buffer[1:]

        self.add_to_stack(subword, pos_tag)

    def take_action_append(self, pos_tag):
        subword = self.buffer[0][1]
        self.buffer = self.buffer[1:]

        self.results[-1]['word'] += subword
        self.results[-1]['pos'] = pos_tag

    def take_action_left_arc(self, dep_label):
        left_node = self.stack[-2]
        right_node = self.stack[-1]

        self.results[left_node]['head_idx'] = right_node
        self.results[left_node]['dep_label'] = dep_label
        self.arcs.append((right_node, dep_label, left_node))

        self.stack[-2] = self.stack[-1]
        self.stack = self.stack[:-1]

    def take_action_right_arc(self, dep_label):
        left_node = self.stack[-2]
        right_node = self.stack[-1]

        self.results[right_node]['head_idx'] = left_node
        self.results[right_node]['dep_label'] = dep_label
        self.arcs.append((left_node, dep_label, right_node))

        self.stack = self.stack[:-1]

    def add_to_stack(self, subword, pos):
        self.stack.append(len(self.results))

        word_info = dict()
        word_info['word_index'] = len(self.results)
        word_info['word'] = subword
        word_info['pos'] = pos
        word_info['head_idx'] = None
        word_info['dep_label'] = None

        self.results.append(word_info)




