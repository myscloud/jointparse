
def get_options(args):
    options = dict()
    for section in args:
        for opt_name in args[section]:
            opt_val = args[section][opt_name]
            options[opt_name] = int(opt_val) if opt_val.isdigit() else opt_val

    return options


def random_func(options):
    # collect_epoch_details('log/zh/pos/model06/', 'log/zh/pos/model06/all_epoch')
    # from tools.preprocess import combine_two_files
    # combine_two_files('results/zh/baseline/pos_results/test.ws', 'results/zh/baseline/parsed/predictBaseline.out',
    #                   'results/zh/baseline/final/zh-test-predict.data')
    # combine_two_files('data/zh/test/zh-ud-test.data', 'results/zh/baseline/parsed/goldBaseline.out',
    #                   'results/zh/baseline/final/zh-test-gold.data')
    # write_subword_file(options)
    # from_specific_to_ud(options)
    write_subword_file(options)


def write_embedding_vectors(options):
    from tools.embedding_reader import NetworkParams
    network_params = NetworkParams()
    network_params.set_word_embedding(options['word_embedding'], options['word_embedding_map'])

    out_file = open('zh-word-emb64.vectors', 'w')
    for word, embedding in zip(network_params.params['word_map'], network_params.params['word_embedding']):
        out_file.write(word + ' ')
        for value in embedding:
            out_file.write(str(value) + ' ')
        out_file.write('\n')

    out_file.close()


def collect_epoch_details(log_dir, output_file_dir):
    from os import listdir
    from os.path import isfile, join

    out_file = open(output_file_dir, 'w')
    n_files = len([f for f in listdir(log_dir) if isfile(join(log_dir, f))])
    for epoch_count in range(0, n_files-1):
        epoch_log_file_dir = join(log_dir, 'epoch_' + str(epoch_count))
        with open(epoch_log_file_dir) as epoch_log_file:
            line = epoch_log_file.read()
            out_file.write(line)

    out_file.close()


def from_specific_to_ud(options):
    from tools.preprocess import specific_to_ud
    specific_to_ud('results/zh/baseline/pos_results/test.ws', 'results/zh/baseline/conll/zh-ud-test.ws')


def write_subword_file(options):
    from tools.preprocess import generate_subword_file
    generate_subword_file(options['baseline_test_file'], options['baseline_test_subword_file'])