
def get_options(args):
    options = dict()
    for section in args:
        for opt_name in args[section]:
            opt_val = args[section][opt_name]
            options[opt_name] = int(opt_val) if opt_val.isdigit() else opt_val

    return options


def random_func(options):
    write_embedding_vectors(options)


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