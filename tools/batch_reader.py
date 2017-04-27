from random import shuffle


class BatchReader:
    def __init__(self, data_dict_list, batch_size):
        """
        :param data_dict_list: a list of a dictionary of {data, pad_element, post_func}
        :param batch_size: an integer indiciating batch size
        """
        self.data_dict_list = data_dict_list
        self.batch_size = batch_size

        # validate
        assert len(self.data_dict_list) > 0
        self.n_data = len(self.data_dict_list[0]['data'])
        for data_dict in self.data_dict_list:
            assert len(data_dict['data']) == self.n_data

        # for batch/epoch count
        self.last_idx = 0
        self.epoch_count = 0
        self.last_epoch_reported = 0

        # for shuffling
        self.shuffle_order = list(range(self.n_data))
        for idx in range(len(self.data_dict_list)):
            self.data_dict_list[idx]['shuffled'] = self.data_dict_list[idx]['data']

    def shuffle(self):
        idx_list = list(range(self.n_data))
        shuffle(idx_list)
        for data_idx in range(len(self.data_dict_list)):
            shuffled_list = list()
            for idx in idx_list:
                shuffled_list.append(self.data_dict_list[data_idx]['data'][idx])
            self.data_dict_list[data_idx]['shuffled'] = shuffled_list

        self.last_idx = 0
        self.shuffle_order = idx_list

    def get_next_batch(self):
        next_idx = self.last_idx + self.batch_size

        batch_data_list = list()
        for data_dict in self.data_dict_list:
            batch_data = self.get_data_next_batch(data_dict['shuffled'],
                                                  self.last_idx, next_idx, data_dict['pad_element'])
            post_func = data_dict['post_func']
            if post_func:
                batch_data = post_func(batch_data)

            batch_data_list.append(batch_data)

        if next_idx > self.n_data:
            self.epoch_count += 1
            next_idx %= self.n_data

        self.last_idx = next_idx
        return batch_data_list

    def reset(self):
        self.shuffle_order = list(range(self.n_data))
        for data_idx in range(len(self.data_dict_list)):
            self.data_dict_list[data_idx]['shuffled'] = self.data_dict_list[data_idx]['data']
        self.last_idx = 0

    def get_data_next_batch(self, data, left, right, pad_element):
        if right < len(data):
            batch_data = data[left:right]
        else:
            if not pad_element:
                if len(data) < self.batch_size:
                    raise Exception
                right %= len(data)
                batch_data = data[left:] + data[0:right]
            else:
                batch_data = data[left:]
                batch_data += [pad_element] * (self.batch_size - len(batch_data))

        return batch_data

    def is_epoch_end(self):
        if self.last_epoch_reported < self.epoch_count:
            self.last_epoch_reported = self.epoch_count
            return True
        return False
