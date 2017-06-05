import configparser
import argparse
import sys

from tools.misc import get_options
from tools.logger import Logger
from tools.misc import random_func
# from pos_tag.train import train as pos_train
from pos_tag.evaluate import evaluate_from_file
from multilang_pos.preprocess import get_log_freq
from multilang_pos.train import train as multilang_pos_train
from multilang_pos.predict import predict as pos_predict
from parse.train import train as parser_train
from parse.predict import predict_data_set
from word_seg.preprocess import prepare_transition_prob
from word_seg.train import train as ws_train
from word_seg.predict import predict as ws_predict
from word_seg.data_prep import write_reformatted_candidates
from tools.global_evaluate import evaluate_baseline

from evaluate.ws_evaluate import evaluate_word_segmentation, evaluate_word_segmentation_coverage

parser = argparse.ArgumentParser(description='Specify an action and config file path')
parser.add_argument('--config', dest='config_file_path', type=str, required=True,
                    help='Path to the configuration file')
parser.add_argument('--train_pos', action='store_true', dest='is_pos_train',
                    help='Train boundary-POS tag on training data')
parser.add_argument('--eval_pos', action='store_true', dest='is_pos_eval',
                    help='Evaluate POS tagging results from a result file')
parser.add_argument('--eval_ws', action='store_true', dest='is_ws_eval',
                    help='Evaluate word segmentation results from a candidate file')
parser.add_argument('--eval_ws_cover', action='store_true', dest='is_ws_eval_cover',
                    help='Evaluate word segmentation coverage')
parser.add_argument('--predict_pos', action='store_true', dest='is_pos_predict',
                    help='Test POS tagging on test data')
parser.add_argument('--test_parser', action='store_true', dest='is_parser_test',
                    help='Test parsing on test data')
parser.add_argument('--train_parser', action='store_true', dest='is_parser_train',
                    help='Train joint parser')
parser.add_argument('--random', action='store_true', dest='is_random',
                    help='Random function, debug mode only :)')
parser.add_argument('--preprocess_pos', action='store_true', dest='is_pos_preprocess',
                    help='Preprocess necessary data for POS tagging')
parser.add_argument('--preprocess_ws', action='store_true', dest='is_ws_preprocess',
                    help='Preprocess necessary data for word segmentation')
parser.add_argument('--train_ws', action='store_true', dest='is_ws_train',
                    help='Train word segmentation on training data')
parser.add_argument('--predict_ws', action='store_true', dest='is_ws_predict',
                    help='Predict word segmentation results and output first k output')
parser.add_argument('--evaluate', action='store_true', dest='is_evaluate',
                    help='Evaluate results from baseline method')

parser_function_map = {
    'is_pos_train': multilang_pos_train,
    'is_pos_predict': pos_predict,
    'is_parser_train': parser_train,
    'is_pos_eval': evaluate_from_file,
    'is_ws_eval': evaluate_word_segmentation,
    'is_ws_eval_cover': evaluate_word_segmentation_coverage,
    'is_parser_test': predict_data_set,
    'is_random': random_func,
    'is_pos_preprocess': get_log_freq,
    'is_ws_preprocess': write_reformatted_candidates,
    'is_ws_train': ws_train,
    'is_ws_predict': ws_predict,
    'is_evaluate': evaluate_baseline
}


def main_process(args):
    # read config file to get options
    config = configparser.ConfigParser()
    config.read(args.config_file_path)
    options = get_options(config)

    # log file
    sys.stdout = Logger(options['log_file_path'])

    # process a function according to the specified command in args
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        if type(arg_value) == bool and arg_value:
            processing_func = parser_function_map[arg_name]
            processing_func(options)
            break


if __name__ == '__main__':
    arguments = parser.parse_args()
    main_process(arguments)
