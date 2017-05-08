from parse.parser import ParserConfig


def left_arc(args, **kwargs):
    dep_label = args[0]
    parser, config = kwargs['parser'], kwargs['config']

    head_node_id = config.stack[-1]
    dependent_node_id = config.stack[-2]
    parser.add_new_arc(head_node_id, dep_label, dependent_node_id, 'left')

    params = {'buffer': config.buffer, 'stack': config.stack}
    new_config = ParserConfig('remove s2', params)
    return new_config


def right_arc(args, **kwargs):
    dep_label = args[0]
    parser, config = kwargs['parser'], kwargs['config']

    head_node_id = config.stack[-2]
    dependent_node_id = config.stack[-1]
    parser.add_new_arc(head_node_id, dep_label, dependent_node_id, 'right')

    params = {'buffer': config.buffer, 'stack': config.stack}
    new_config = ParserConfig('remove s1', params)
    return new_config


def shift(args, **kwargs):
    pos_tag = args[0]
    parser, config = kwargs['parser'], kwargs['config']

    shifted_node = config.buffer[0]
    new_stack_index = parser.add_new_stack_node(shifted_node, pos_tag)

    params = {'buffer': config.buffer, 'stack': config.stack, 'new_stack_index': new_stack_index}
    new_config = ParserConfig('shift', params)
    return new_config


def append(args, **kwargs):
    pos_tag = args[0]
    parser, config = kwargs['parser'], kwargs['config']

    appended_node = config.buffer[0]
    tos_node = config.stack[-1]
    parser.append_to_top_of_stack(tos_node, appended_node, pos_tag)

    params = {'buffer': config.buffer, 'stack': config.stack}
    new_config = ParserConfig('remove b1', params)
    return new_config

action_dict = {
    'LEFT-ARC': left_arc,
    'RIGHT-ARC': right_arc,
    'SHIFT': shift,
    'APPEND': append
}


def take_parser_action_from_index(parser, action_index, reverse_action_map, action_list):
    action_tuple = reverse_action_map[action_index]
    (action, params) = action_tuple
    _, feasible_actions = get_feasible_actions_with_list(parser, action_list)
    if action not in feasible_actions:
        print(action_index, feasible_actions)
        raise Exception
    take_parser_action(parser, action_tuple)


def take_parser_action(parser, action_tuple):
    action, args = action_tuple[0], list(action_tuple[1:])
    config = parser.current_config
    parser.actions.append(action_tuple)
    action_dict[action](args, parser=parser, config=config)


def get_feasible_actions(parser, action_list):
    feasibility_vec, _ = get_feasible_actions_with_list(parser, action_list)
    return feasibility_vec


def get_feasible_actions_with_list(parser, action_list):
    feasibility = list()
    feasible_actions = list()
    config = parser.current_config
    last_action = parser.actions[-1][0] if len(parser.actions) > 0 else None

    for (action, start_index, end_index) in action_list:
        feasible = False

        if action == 'LEFT-ARC':
            feasible = (len(config.stack) >= 2) and (config.stack[-2] != 'word0')
        elif action == 'RIGHT-ARC':
            feasible = (len(config.stack) >= 2)
        elif action == 'SHIFT':
            feasible = (len(config.buffer) > 0)
        elif action == 'APPEND':
            feasible = (len(config.buffer) > 0) and (last_action == 'SHIFT' or last_action == 'APPEND')

        feasibility += [float(feasible)] * (end_index - start_index)
        if feasible:
            feasible_actions.append(action)

    return feasibility, feasible_actions


def get_action_index(action_map):
    action_index_list = list()
    current_action = ''
    start_index = None
    for action_idx, (action, params) in enumerate(action_map):
        if action != current_action:
            if start_index is not None:
                action_index_list.append((current_action, start_index, action_idx))

            start_index = action_idx
            current_action = action

    if start_index is not None:
        action_index_list.append((current_action, start_index, len(action_map)))

    return action_index_list
