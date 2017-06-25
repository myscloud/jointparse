
def specific_to_ud(file_path, out_file_path):
    out_file = open(out_file_path, 'w')
    sent_count = 0
    with open(file_path) as input_file:
        for line in input_file:
            if len(line) > 1 and line[0] != '#':
                sent_count = 1
                tokens = line.strip().split('\t')
                out_file.write(tokens[0] + '\t' + tokens[1] + '\t_\t' + tokens[2] + '\t' + tokens[2] +
                               '\t_\t' + tokens[3] + '\t'
                               + tokens[4] + '\t_\t_\n')

            if line[0] == '#' and sent_count > 0:
                out_file.write('\n')


def combine_two_files(before_file, after_file, output_file_path):
    """
    Just for combine results from SOTA process (LSTM parser) back for evaluation
    """
    def get_token_list(file_name):
        sentence_list = list()
        curr_sentence = list()
        with open(file_name) as data_file:
            for line in data_file:
                if len(line) > 1 and line[0] != '#':
                    tokens = line.strip().split('\t')
                    curr_sentence.append(tokens)
                elif len(line) <= 1 and len(curr_sentence) > 0:
                    sentence_list.append(curr_sentence)
                    curr_sentence = list()

            if len(curr_sentence) > 0:
                sentence_list.append(curr_sentence)

        return sentence_list

    before_list = get_token_list(before_file)
    after_list = get_token_list(after_file)

    with open(output_file_path, 'w') as out_file:
        sent_count = 1
        for before_sent, after_sent in zip(before_list, after_list):
            out_file.write('#Sentence ' + str(sent_count) + '\n')
            for before_tok, after_tok in zip(before_sent, after_sent):
                out_file.write('\t'.join(before_tok[0:3]) + '\t' + '\t'.join(after_tok[6:8]) + '\n')
            out_file.write('\n')
            sent_count += 1


def get_char_subword(word):
    return list(word)


def generate_subword_file(data_file_path, subword_out_file, subword_func=get_char_subword):
    from tools.experiment_data import ExperimentData
    sent_data = ExperimentData.read_data(data_file_path)

    with open(subword_out_file, 'w') as subword_file:
        for sent_idx, sentence in enumerate(sent_data):
            subword_file.write('#Sentence ' + str(sent_idx + 1) + '\n')
            subword_count = 0

            for word_idx, word_info in enumerate(sentence):
                subwords = subword_func(word_info.word)
                for subword in subwords:
                    subword_count += 1
                    subword_file.write(str(subword_count) + '\t' + subword + '\t' + str(word_idx + 1) + '\n')

            subword_file.write('\n')
