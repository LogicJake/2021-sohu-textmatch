import copy
import gc
import json
import logging
import os
import random

import numpy as np
import torch.utils.data as Data
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InputFeatures(object):
    def __init__(self,
                 task_type,
                 input_ids,
                 attention_mask=None,
                 token_type_ids=None,
                 label=None,
                 sentence_w2v_embedding0=None,
                 sentence_w2v_embedding1=None):
        self.task_type = task_type
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    
def convert2features(examples,
                     tokenizer,
                     max_length=512,
                     pad_token=0,
                     pad_token_segment_id=0):
    features = []
    for example in tqdm(examples):
        inputs = tokenizer.encode_plus(example[1],
                                       example[2],
                                       add_special_tokens=True,
                                       max_length=max_length,
                                       truncation=True)
        input_ids, token_type_ids = inputs['input_ids'], inputs[
            'token_type_ids']
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] *
                                           padding_length)

        if example[3] is not None:
            label = example[3]
        else:
            label = 0

        features.append(
            InputFeatures(example[0], input_ids, attention_mask,
                          token_type_ids, label))

    return features


class BuildDataSet(Data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        task_type = np.array(feature.task_type)
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        label = np.array(feature.label)

        return task_type, input_ids, attention_mask, token_type_ids, label

    def __len__(self):
        return len(self.features)


class DataProcessor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.seed = config.seed
        self.variants = config.variants
        self.stop_word_list = None

    def get_train_examples(self):
        train_examples = []

        for i, var in enumerate(self.variants):
            key = 'labelA' if 'A' in var else 'labelB'
            fs = [
                os.path.join(self.data_dir, 'sohu2021_open_data', var,
                             'train.txt'),
                os.path.join(self.data_dir, 'round2', f'{var}.txt'),
                os.path.join(self.data_dir, 'divided_20210419', var, 'train.txt'),
            ]

            for f in fs:
                with open(f) as f:
#                     for line in list(f)[:10]:
                    for line in f:
                        line = json.loads(line)
                        train_examples.append((i, line['source'],
                                               line['target'], int(line[key])))

        return train_examples

    def get_valid_examples(self):
        valid_examples = []

        for i, var in enumerate(self.variants):
            key = 'labelA' if 'A' in var else 'labelB'
            f = os.path.join(self.data_dir, 'sohu2021_open_data', var,
                             'valid.txt')

            with open(f) as f:
                for line in f:
                    line = json.loads(line)
                    valid_examples.append(
                        (i, line['source'], line['target'], int(line[key])))
#                     break

        return valid_examples

    def get_test_examples(self):
        test_ids = []
        test_examples = []

        for i, var in enumerate(self.variants):
            f = os.path.join(self.data_dir, 'sohu2021_open_data', var,
                             'test_with_id.txt')

            with open(f) as f:
                for line in f:
                    line = json.loads(line)
                    test_examples.append(
                        (i, line['source'], line['target'], -1))
                    test_ids.append(line['id'])

        return test_ids, test_examples

    def read_data_augment(self, augment_list):
        data_augment = []

        for augment in augment_list:
            for type in self.data_files:
                examples = self._read_data(
                    os.path.join('data/augment_data', type,
                                 '{}.pkl'.format(augment)))
                data_augment += examples

        random.seed(self.seed)
        random.shuffle(data_augment)
        return data_augment
