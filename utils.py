import copy
import json

import numpy as np
import torch


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_to_dict(config):
    output = copy.deepcopy(config.__dict__)
    output['device'] = config.device.type
    return output


def config_to_json_string(config):
    return json.dumps(config_to_dict(config), indent=2, sort_keys=True)


def combined_result(all_result, weight=None, pattern='average'):
    def average_result(all_result):  # shape:[num_model, axis]
        all_result = np.asarray(all_result, dtype=np.float)
        return np.mean(all_result, axis=0)

    def weighted_result(all_result, weight):
        all_result = np.asarray(all_result, dtype=np.float)
        return np.average(all_result, axis=0, weights=weight)

    if pattern == 'weighted':
        return weighted_result(all_result, weight)
    elif pattern == 'average':
        return average_result(all_result)
    else:
        raise ValueError("the combined type is incorrect")

def sentence_reverse(test_examples):
    """
    将测试数据翻转
    :param test_examples:
    :return:
    """
    reverse_test_examples = []
    for example in test_examples:
        try_example = [example[1], example[0], example[2]]
        reverse_test_examples.append(try_example)
    return reverse_test_examples
