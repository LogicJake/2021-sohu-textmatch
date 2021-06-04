import logging
import os

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer

from data_processor import BuildDataSet, convert2features
from train_valid import model_evaluate, model_save, model_train

logger = logging.getLogger(__name__)


def hold_out(config, model, train_data, valid_data, train_enhancement=None):
    logger.debug('训练集维度：{}，验证集维度：{}'.format(len(train_data), len(valid_data)))
    # 数据增强
    if train_enhancement:
        logger.debug('通过数据增强后，新增数据: %d', len(train_enhancement))
        train_data.extend(train_enhancement)

    # 读取 Tokenizer
    if config.pretrain_model_name == 'albert_chinese_large':
        tokenizer = BertTokenizer.from_pretrained(
            config.pretrain_path, do_lower_case=config.do_lower_case)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrain_path, do_lower_case=config.do_lower_case)

    # 训练集数据加载
    train_features = convert2features(examples=train_data,
                                      tokenizer=tokenizer,
                                      max_length=config.pad_size)
    train_dataset = BuildDataSet(train_features)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)

    # 验证集数据加载
    valid_features = convert2features(examples=valid_data,
                                      tokenizer=tokenizer,
                                      max_length=config.pad_size)
    valid_dataset = BuildDataSet(valid_features)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)

    # 模型训练保存
    model = model.to(config.device)
    best_model = model_train(config, model, train_loader, valid_loader)

    metrics, valid_loss, total_inputs_error = model_evaluate(
        config, best_model, valid_loader)

    valid_f1 = metrics['f1']
    valid_f1_a = metrics['f1_a']
    valid_f1_b = metrics['f1_b']

    logger.info(
        'evaluate: f1: {0:>6.2%}, f1_a: {1:>6.2%}, f1_b: {2:>6.2%}, loss: {3:>.6f}'
        .format(valid_f1, valid_f1_a, valid_f1_b, valid_loss))
    model_save(config, best_model)

    # 保存 bad_case
    for example in total_inputs_error:
        tokens = tokenizer.convert_ids_to_tokens(example['sentence_ids'])
        if config.pretrain_model_name == 'chinese-xlnet-base':
            tokens = ''.join(x for x in tokens if x not in ['<cls>', '[pad]'])
            source, target, _ = tokens.split('<sep>')
        else:
            tokens = ''.join(x for x in tokens if x not in ['[CLS]', '[PAD]'])
            source, target, _ = tokens.split('[SEP]')
        example['source'] = source
        example['target'] = target

    bad_case = pd.DataFrame(total_inputs_error)

    os.makedirs('user_data/bad_case', exist_ok=True)
    bad_case.to_csv('user_data/bad_case/{}.csv'.format(config.model_name),
                    index=False)
