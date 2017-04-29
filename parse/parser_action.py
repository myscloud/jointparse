from parse.parser import ParserConfig


def left_arc(parser, config, dep_label):
    head_node_id = config.stack[-1]
    dependent_node_id = config.stack[-2]
    parser.add_new_arc(head_node_id, dep_label, dependent_node_id, 'left')
    new_config = ParserConfig('remove s2', config)
    return new_config