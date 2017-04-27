
class FeaturePattern:
    def __init__(self, feature_file_path, network_params=None):
        self.feature_dict = FeaturePattern.read_feature_file(feature_file_path)
        self.node_list, self.relative_node_list = self.get_statistics()
        if network_params:
            self.none_index = FeaturePattern.generate_none_index(network_params)

    def get_statistics(self):
        node_list = set()
        relative_node_list = set()

        for category_name in self.feature_dict:
            for (node_name, _) in self.feature_dict[category_name]:
                if ':' in node_name:
                    relative_node_list.add(node_name)
                else:
                    node_list.add(node_name)

        return list(node_list), list(relative_node_list)

    @staticmethod
    def generate_none_index(network_params):
        none_dict = dict()
        none_dict['word'] = network_params.params['word_map']['<PAD>']
        none_dict['pos'] = network_params.params['pos_map']['<PAD>']
        none_dict['label'] = network_params.params['dep_label_map']['<PAD>']
        none_dict['bpos'] = network_params.params['bpos_map']['<PAD>']
        none_dict['subword'] = network_params.params['subword_map']['<PAD>']
        return none_dict

    @staticmethod
    def read_feature_file(feature_file_path):
        feature_dict = dict()
        latest_feature_name = None

        with open(feature_file_path) as feature_file:
            for line in feature_file:
                if line[0] == '[':
                    latest_feature_name = line[1:line.index(']')]
                    feature_dict[latest_feature_name] = list()
                else:
                    features = line.strip().split(' ')
                    for feature in features:
                        feature_dict[latest_feature_name].append(tuple(feature.split('.')))

        return feature_dict

