
class DataPOS:
    def __init__(self, subwords, lang_params, pos_type, train_sent_len=30, overlap_train=2, data_prop=None):
        self.n_len = train_sent_len
        self.overlap = overlap_train

        if pos_type == 'pos':
            self.words, word_special_tok, words_index, pos_special_tok, pos_index, self.pos_val = \
                DataPOS.get_word_essential_data(data_prop, lang_params)
        else:  # bpos
            self.words, word_special_tok, words_index, pos_special_tok, pos_index, self.pos_val = \
                DataPOS.get_subword_essential_data(subwords, lang_params, data_prop)

        self.predicted_tags = None

        self.phrases, self.sent_phrase_map, self.phrase_token_map, self.phrase_len = \
            self.trim_sentence(words_index, word_special_tok)
        if pos_index is not None:
            self.phrases_label, _, _, _ = self.trim_sentence(pos_index, pos_special_tok)

    @staticmethod
    def get_subword_essential_data(subwords_info, lang_params, data_prop):
        subwords = DataPOS.get_subword_word(subwords_info)
        subword_special_tok = {'START': lang_params.params['subword_map']['<S>'],
                            'END': lang_params.params['subword_map']['</S>'],
                            'PAD': lang_params.params['subword_map']['<PAD>']}
        subwords_index = lang_params.map_with_params(subwords, 'subword_map', unk_name='<UNK>')

        pos_special_tok, pos_index, pos_val = None, None, None
        if data_prop:
            single_x = lang_params.params['bpos_map']['S-X']
            pos_special_tok = {'START': single_x, 'END': single_x, 'PAD': single_x}

            pos_val = DataPOS.generate_boundary_pos_tag(subwords_info, data_prop)
            pos_index = lang_params.map_with_params(pos_val, 'bpos_map')

        return subwords, subword_special_tok, subwords_index, pos_special_tok, pos_index, pos_val

    @staticmethod
    def get_word_essential_data(data_prop, lang_params):
        words = list()
        pos_list = list()
        for sentence in data_prop:
            word_sentence = [x.word for x in sentence]
            words.append(word_sentence)

            pos_sentence = [x.pos for x in sentence]
            pos_list.append(pos_sentence)

        word_special_tok = {'START': lang_params.params['word_map']['<S>'],
                            'END': lang_params.params['word_map']['</S>'],
                            'PAD': lang_params.params['word_map']['<PAD>']}
        words_index = lang_params.map_with_params(words, 'word_map', unk_name='<UNK>')

        pos_x = lang_params.params['pos_map']['X']
        pos_special_tok = {'START': pos_x, 'END': pos_x, 'PAD': pos_x}
        pos_index = lang_params.map_with_params(pos_list, 'pos_map')

        return words, word_special_tok, words_index, pos_special_tok, pos_index, pos_list

    @staticmethod
    def get_subword_word(subwords_info):
        subwords = list()
        for sentence in subwords_info:
            subword_sentence = [x.subword for x in sentence]
            subwords.append(subword_sentence)

        return subwords

    @staticmethod
    def generate_boundary_pos_tag(subwords_info, data_prop):
        pos_list = list()
        for subword_sent, data_sent in zip(subwords_info, data_prop):
            sentence_pos_list = list()
            for subword_idx, subword_info in enumerate(subword_sent):
                left_word_idx = subword_sent[subword_idx-1].word_idx if subword_idx > 0 else None
                curr_word_idx = subword_sent[subword_idx].word_idx
                right_word_idx = subword_sent[subword_idx+1].word_idx if subword_idx < len(subword_sent) - 1 \
                    else None

                boundary_type = DataPOS.get_boundary_type(left_word_idx, curr_word_idx, right_word_idx)
                boundary_pos_tag = boundary_type + '-' + data_sent[curr_word_idx-1].pos
                sentence_pos_list.append(boundary_pos_tag)
            pos_list.append(sentence_pos_list)

        return pos_list

    def trim_sentence(self, data_list, special_tok_dict):
        """
        :param data_list:
        :param special_tok_dict:
        :return:
         - a list of phrases, each phrase has size of self.n_len, if the original sentence (in data_list)
         is longer than self.n_len - 2 (2 for START,END), we will repeat last |self.overlap| characters
         - sent_phrase_map: tuple of (start_phrase_index, end_phrase_index) of each sentence
         - phrase_token_map: tuple of (start, end) index of data to be evaluated
        """
        assert self.n_len > self.overlap
        trimmed_phrases = list()
        sent_phrase_map = list()
        phrase_token_map = list()
        phrase_count = 0
        phrase_len_list = list()

        for sentence in data_list:
            last_idx = 0
            start_phrase_idx = phrase_count
            end_tok_placed = False

            while not end_tok_placed:
                curr_phrase = list()
                if last_idx == 0:
                    curr_phrase.append(special_tok_dict['START'])
                    phrase_start_idx = 1
                else:
                    curr_phrase.extend(sentence[last_idx-self.overlap:last_idx])
                    phrase_start_idx = self.overlap

                next_idx = min(len(sentence), last_idx + (self.n_len - len(curr_phrase)))
                curr_phrase.extend(sentence[last_idx:next_idx])

                if len(curr_phrase) < self.n_len:
                    # starting phrase, ending phrase, ending index
                    phrase_end_idx = len(curr_phrase)

                    curr_phrase.append(special_tok_dict['END'])
                    end_tok_placed = True
                    sent_phrase_map.append((start_phrase_idx, phrase_count))
                    phrase_len = len(curr_phrase)

                    curr_phrase.extend([special_tok_dict['PAD']] * (self.n_len - len(curr_phrase)))
                else:
                    phrase_len = self.n_len
                    phrase_end_idx = self.n_len

                trimmed_phrases.append(curr_phrase)
                phrase_len_list.append(phrase_len)
                phrase_token_map.append((phrase_start_idx, phrase_end_idx))
                last_idx = next_idx
                phrase_count += 1

        return trimmed_phrases, sent_phrase_map, phrase_token_map, phrase_len_list

    def map_results_back(self, predicted_labels, reverse_map):
        predicted_results = list()
        for (start_phrase_idx, end_phrase_idx) in self.sent_phrase_map:
            sentence_tags = list()
            for phrase_idx in range(start_phrase_idx, end_phrase_idx + 1):
                left_idx, right_idx = self.phrase_token_map[phrase_idx]
                for subword_idx in range(left_idx, right_idx):
                    tag_labels = predicted_labels[phrase_idx][subword_idx]
                    sentence_tags.append([reverse_map[label] for label in tag_labels])

            predicted_results.append(sentence_tags)

        self.predicted_tags = predicted_results
        return predicted_results

    def write_tagged_results(self, output_path):
        out_file = open(output_path, 'w')
        sentence_count = 0

        for sentence_info, sentence_tags, sent_real_tag in zip(self.words, self.predicted_tags, self.pos_val):
            out_file.write('#Sentence ' + str(sentence_count + 1) + '\n')
            subword_count = 0
            for word, pos_tags, real_tag in zip(sentence_info, sentence_tags, sent_real_tag):
                candidate_tags = ','.join(pos_tags)
                out_file.write(str(subword_count) + '\t' + word + '\t' + candidate_tags + '\t' + real_tag + '\n')
                subword_count += 1

            out_file.write('\n')
            sentence_count += 1

        out_file.close()

    @staticmethod
    def get_boundary_type(left, current, right):
        if (current == right) and (not left or left != current):
            return 'B'
        elif (left == current) and (not right or current != right):
            return 'E'
        elif left and right and left == right:
            return 'I'
        else:
            return 'S'
