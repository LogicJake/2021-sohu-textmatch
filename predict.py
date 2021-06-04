import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert import Bert
from data_processor import BuildDataSet, DataProcessor, convert2features
from train_valid import model_evaluate, model_load
from utils import combined_result, random_seed


class ModelConfig():
    def __init__(self):
        pass


class TestConfig:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.data_dir = 'raw_data'.replace('/', os.path.sep)

        self.model_path = 'user_data/model'.replace('/', os.path.sep)
        self.model_names = ['erniev8']
        self.variants = [
            '短短匹配A类',
            '短短匹配B类',
            '短长匹配A类',
            '短长匹配B类',
            '长长匹配A类',
            '长长匹配B类',
        ]

        os.makedirs('result', exist_ok=True)
        self.output_path = os.path.join(
            'result',
            time.strftime('%Y-%m-%d_%H-%M-%S') + '.csv')

        self.prob_threshold = 0.5
        self.seed = 2021


def model_predict(model, config, examples):
    # 读取 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrain_path, do_lower_case=config.do_lower_case)

    test_features = convert2features(examples=examples,
                                     tokenizer=tokenizer,
                                     max_length=config.pad_size)
    test_dataset = BuildDataSet(test_features)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False)
    test_prob, _ = model_evaluate(config, model, test_loader, test=True)
    return test_prob


def predict_task(config):
    processor = DataProcessor(config)

    ids, examples = processor.get_test_examples()
    all_predict = []

    for model_name in config.model_names:
        with open(os.path.join(config.model_path, model_name, 'config.json'),
                  'r') as f:
            model_conf_json = json.load(f)

        # 模型定义
        model_conf = ModelConfig()
        model_conf.__dict__.update(model_conf_json)
        model = Bert(model_conf)

        # 加载模型
        model_path = os.path.join(config.model_path, model_name)
        print(f'load model from {model_path}')
        model = model_load(model_path, model, device='cpu')
        model.to(config.device)

        # 模型预测
        predict_prob = model_predict(model, model_conf, examples)
        all_predict.append(predict_prob)

    final_predict = combined_result(all_predict, pattern='average')
    final_predict_label = np.asarray(final_predict >= config.prob_threshold,
                                     dtype=np.int)

    submit = pd.DataFrame()
    submit['id'] = ids
    submit['label'] = final_predict_label
    return submit


def predict(config):
    start_time = time.time()

    submit = predict_task(config)

    end_time = time.time()
    time_dif = end_time - start_time
    print(time_dif * 1000 // (14913 + 14909))

    submit.to_csv(config.output_path, index=False)


if __name__ == '__main__':
    config = TestConfig()
    random_seed(config.seed)
    predict(config)
