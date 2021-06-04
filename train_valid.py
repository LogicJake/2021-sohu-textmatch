# coding: UTF-8
import copy
import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import config_to_json_string

logger = logging.getLogger(__name__)


class FGM():
    '''
    Example
    # 初始化
    fgm = FGM(model,epsilon=1, emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def model_train(config, model, train_iter, valid_iter):
    start_time = time.time()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    diff_part = ["bert.embeddings", "bert.encoder"]
    if not config.diff_learning_rate:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate)
    else:
        logger.info("use the diff learning rate")
        # the formal is basic_bert part, not include the pooler
        optimizer_grouped_parameters = [
            {
                # weight 衰减
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n
                               for nd in no_decay) and any(nd in n
                                                           for nd in diff_part)
                ],
                "weight_decay":
                config.weight_decay,
                "lr":
                config.learning_rate
            },
            {
                # weight 不衰减
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n
                           for nd in no_decay) and any(nd in n
                                                       for nd in diff_part)
                ],
                "weight_decay":
                0.0,
                "lr":
                config.learning_rate
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(
                        nd in n for nd in diff_part)
                ],
                "weight_decay":
                config.weight_decay,
                "lr":
                config.head_learning_rate
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n
                           for nd in no_decay) and not any(nd in n
                                                           for nd in diff_part)
                ],
                "weight_decay":
                0.0,
                "lr":
                config.head_learning_rate
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    t_total = len(train_iter) * config.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=t_total *
                                                config.warmup_proportion,
                                                num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s", config.device)

    global_batch = 0  # 记录进行到多少batch
    valid_best_f1 = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    predict_all = []
    labels_all = []
    best_model = copy.deepcopy(model)

    if config.FGM:
        fgm = FGM(model, epsilon=1, emb_name='word_embeddings.')

    for epoch in range(config.num_train_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))
        for _, (task_type, input_ids, attention_mask, token_type_ids, labels) in enumerate(train_iter):
            global_batch += 1
            model.train()

            task_type = torch.tensor(task_type).type(torch.LongTensor).to(
                config.device)
            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(
                config.device)
            attention_mask = torch.tensor(attention_mask).type(
                torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(
                torch.LongTensor).to(config.device)

            if config.loss_method in ['binary']:
                labels_tensor = torch.tensor(labels).type(
                    torch.FloatTensor).to(config.device)
            else:
                labels_tensor = torch.tensor(labels).type(torch.LongTensor).to(
                    config.device)

            outputs, loss = model(task_type, input_ids, attention_mask,
                                  token_type_ids, labels_tensor)

            loss.backward()

            # 对抗训练
            if config.FGM:
                fgm.attack()  # 在embedding上添加对抗扰动
                _, loss_adv = model(task_type, input_ids, attention_mask, token_type_ids,
                                    labels_tensor)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            outputs = outputs.cpu().detach().numpy()
            predic = list(
                np.array(outputs >= config.prob_threshold, dtype='int'))
            labels_all.extend(labels)
            predict_all.extend(predic)

            if global_batch % 100 == 0:
                train_f1 = f1_score(labels_all, predict_all)
                predict_all = []
                labels_all = []

                metrics, valid_loss, _ = model_evaluate(
                    config, model, valid_iter)
                valid_f1 = metrics['f1']
                valid_f1_a = metrics['f1_a']
                valid_f1_b = metrics['f1_b']

                if valid_f1 > valid_best_f1:
                    valid_best_f1 = valid_f1
                    improve = '*'
                    last_improve = global_batch
                    best_model = copy.deepcopy(model)
                else:
                    improve = ''

                time_dif = time.time() - start_time
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6f},  Train F1: {2:>6.2%},  Val Loss: {3:>5.6f},  Val F1: {4:>6.2%}, Val F1_a: {5:>6.2%}, Val F1_b: {6:>6.2%}, Time: {7} {8}'
                logger.info(
                    msg.format(global_batch,
                               loss.cpu().data.item(), train_f1,
                               valid_loss.cpu().data.item(), valid_f1,
                               valid_f1_a, valid_f1_b, time_dif, improve))

            if config.early_stop and global_batch - last_improve > config.require_improvement:
                logger.info(
                    "No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    
    if config.early_stop:
        return best_model
    else:
        return model

def model_evaluate(config, model, data_iter, test=False):
    model.eval()

    # loss 总和
    loss_total = 0
    # 预测的全部 label
    predict_label_all = []
    predict_label_taskA = []
    predict_label_taskB = []

    # 预测的全部概率
    predict_prob_all = []

    # 真实的全部 label
    true_label_all = []
    true_label_taskA = []
    true_label_taskB = []

    # 全部的task_type
    task_type_all = []

    total_inputs_error = []
    with torch.no_grad():
        for i, (task_type, input_ids, attention_mask, token_type_ids, labels) in enumerate(data_iter):

            task_type = torch.tensor(task_type).type(torch.LongTensor).to(
                config.device)
            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(
                config.device)
            attention_mask = torch.tensor(attention_mask).type(
                torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(
                torch.LongTensor).to(config.device)

            if config.loss_method in ['binary']:
                labels = torch.tensor(labels).type(torch.FloatTensor).to(
                    config.device) if not test else None
            else:
                labels = torch.tensor(labels).type(torch.LongTensor).to(
                    config.device) if not test else None

            predict_prob, loss = model(task_type, input_ids, attention_mask,
                                       token_type_ids, labels)

            predict_prob = predict_prob.cpu().detach().numpy()
            predict_label = list(
                np.array(predict_prob >= config.prob_threshold, dtype='int'))

            predict_prob_all.extend(list(predict_prob))
            predict_label_all.extend(predict_label)
            task_type_all.extend(list(task_type.cpu().detach().numpy()))

            if not test:
                labels = labels.data.cpu().numpy()
                true_label_all.extend(list(labels))
                loss_total += loss

                input_ids = input_ids.data.cpu().detach().numpy()
                classify_error = get_classify_error(input_ids, predict_label,
                                                    labels, predict_prob)
                total_inputs_error.extend(classify_error)

    if test:
        return predict_prob_all, predict_label_all

    for task_type, predict_label, true_label in zip(task_type_all,
                                                    predict_label_all,
                                                    true_label_all):
        if task_type % 2 == 0:
            predict_label_taskA.append(predict_label)
            true_label_taskA.append(true_label)
        else:
            predict_label_taskB.append(predict_label)
            true_label_taskB.append(true_label)

    f1_a = f1_score(true_label_taskA, predict_label_taskA)
    f1_b = f1_score(true_label_taskB, predict_label_taskB)
    f1 = (f1_a + f1_b) / 2

    return {
        'f1': f1,
        'f1_a': f1_a,
        'f1_b': f1_b
    }, loss_total / len(data_iter), total_inputs_error


def get_classify_error(input_ids, predict, labels, proba, input_ids_pair=None):
    error_list = []
    error_idx = predict != labels
    error_sentences = input_ids[error_idx]
    total_sentences = []
    if input_ids_pair is not None:
        error_sentences_pair = input_ids_pair[error_idx]
        for sentence1, sentence2 in zip(error_sentences, error_sentences_pair):
            total_sentences.append(
                np.array(sentence1.tolist() + [117] + sentence2.tolist(),
                         dtype=int))
    else:
        total_sentences = error_sentences

    true_label = labels[error_idx]
    pred_proba = proba[error_idx]
    for sentences, label, prob in zip(total_sentences, true_label, pred_proba):
        error_dict = {}
        error_dict['sentence_ids'] = sentences
        error_dict['true_label'] = label
        error_dict['proba'] = prob
        error_list.append(error_dict)

    return error_list


def model_save(config, model, num=-1):
    if num == -1:
        file_name = os.path.join(config.model_path, 'model.pkl')

    with open(os.path.join(config.model_path, 'config.json'), 'w') as f:
        f.write(config_to_json_string(config))

    torch.save(model.state_dict(), file_name)
    logger.info('model saved, path: %s', file_name)


def model_load(model_path, model, device='cpu'):
    file_name = os.path.join(model_path, 'model.pkl')
    model.load_state_dict(
        torch.load(file_name,
                   map_location=device if device == 'cpu' else "{}:{}".format(
                       device, 0)))
    return model
