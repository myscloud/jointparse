import numpy as np


def to_matrix(data):
    vector_list = list()
    for sentence in data:
        vector_list.append(np.asarray(sentence))
    return np.vstack(tuple(vector_list))


def to_tensor(data):
    return np.array(data)


def map_shuffled_list_back(data_list, shuffle_order):
    original_list = [None] * len(shuffle_order)
    for data, original_idx in zip(data_list, shuffle_order):
        original_list[original_idx] = data

    return original_list


def fill_right_to_tensor(data, pad_number):
    """
    :param data:
    :param pad_number:
    :return: a tensor with equal-sized data (padded unequal columns with pad_number)
    """
    max_len = max([len(sentence) for sentence in data])
    padded_data = list()
    for sentence in data:
        padded_sentence = sentence + ([pad_number] * (max_len - len(sentence)))
        padded_data.append(padded_sentence)

    return to_tensor(padded_data)


def argmax_max_k(data, k_max):
    index_sort = np.argsort(data, axis=-1)
    return index_sort[:, :, -1:-(k_max+1):-1]  # get k-max element (last element to -k_max index in the sorted vec)
