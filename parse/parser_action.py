from parse.parser import ParserConfig


def left_arc(parser, config, dep_label):

    new_config = ParserConfig('remove s2', config)