import logging
import os
import time
import warnings

import torch

from bert import Bert
from cross_validation import hold_out
from data_processor import DataProcessor
from utils import config_to_json_string, random_seed

warnings.filterwarnings('ignore')


class TrainConfig:
    def __init__(self):
        # 预训练模型相关
        self.pretrain_model_name = 'chinese-roberta-wwm-ext'
        self.pretrain_path = 'data/pretrain_models/{}'.format(
            self.pretrain_model_name).replace('/', os.path.sep)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.do_lower_case = True
        self.requires_grad = True

        # 模型相关
        self.pad_size = 512  # 每句话处理成的长度
        self.batch_size = 32
        self.learning_rate = 2e-5  # 学习率
        self.head_learning_rate = 1e-4  # 后面的分类层的学习率
        self.weight_decay = 0.01  # 权重衰减因子
        self.warmup_proportion = 0.1  # Proportion of training to perform linear learning rate warmup for.
        self.num_train_epochs = 3  # epoch数
        self.prob_threshold = 0.5
        self.loss_method = 'binary'  # [ binary, cross_entropy]
        self.hidden_dropout_prob = 0.1
        self.hidden_size = []
        self.diff_learning_rate = False
        self.early_stop = True
        self.require_improvement = 3000
        self.FGM = True
        self.pool_method = 'mean'
        self.multi_drop = 1

        self.variants = [
            '短短匹配A类',
            '短短匹配B类',
            '短长匹配A类',
            '短长匹配B类',
            '长长匹配A类',
            '长长匹配B类',
        ]

        # 数据路径
        self.data_dir = 'raw_data'.replace('/', os.path.sep)
        self.model_name = f'{self.pretrain_model_name}v2'
        self.model_path = 'user_data/model/{}'.format(self.model_name).replace(
            '/', os.path.sep)
        os.makedirs(self.model_path, exist_ok=True)

        # logging
        self.logging_dir = 'user_data/logging/{}'.format(
            self.model_name).replace('/', os.path.sep)
        os.makedirs(self.logging_dir, exist_ok=True)
        self.seed = 2021

        # 数据增强
        self.data_augment = None


def train_model(config):
    logging.debug('config {}'.format(config_to_json_string(config)))

    # 读取数据
    processor = DataProcessor(config)
    train_examples = processor.get_train_examples()
    valid_examples = processor.get_valid_examples()

    if config.data_augment:
        augment_examples = processor.read_data_augment(config.data_augment)
    else:
        augment_examples = None

    logging.info(train_examples[:1])
    logging.info(valid_examples[:1])

    model = Bert(config)
    hold_out(config=config,
             model=model,
             train_data=train_examples,
             valid_data=valid_examples,
             train_enhancement=augment_examples)


if __name__ == '__main__':
    config = TrainConfig()

    random_seed(config.seed)

    # 定义日志
    file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
    logging_filename = os.path.join(config.logging_dir, file)
    logging.basicConfig(filename=logging_filename,
                        format='%(levelname)s: %(message)s',
                        level=logging.DEBUG)

    # 运行模型
    train_model(config)
