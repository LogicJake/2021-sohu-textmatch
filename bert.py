import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1 - pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


def compute_loss(outputs, labels, loss_method='binary'):
    loss = 0.
    if loss_method == 'binary':
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels)
    elif loss_method == 'cross_entropy':
        loss = F.cross_entropy(outputs, labels)
    elif loss_method == 'focal_loss':
        focal_loss = FocalLoss()
        loss = focal_loss(outputs, labels)
    else:
        raise Exception('loss_method {binary or cross_entropy} error. ')
    return loss


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()

        # 计算loss的方法
        self.loss_method = config.loss_method
        self.pool_method = config.pool_method

        self.bert = AutoModel.from_pretrained(config.pretrain_path)

        if config.requires_grad:
            for param in self.bert.parameters():
                param.requires_grad = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        last_layer_dim = self.bert.config.hidden_size

        self.task_type_embedding = nn.Embedding(2, last_layer_dim)
        self.w2v_linear = nn.Linear(200, last_layer_dim)

        self.ln = torch.nn.LayerNorm(last_layer_dim)

        hidden_size = [last_layer_dim] + copy.deepcopy(config.hidden_size)

        self.classifier = nn.Sequential()

        for i in range(len(hidden_size) - 1):
            self.classifier.add_module(
                'classifier_{}'.format(i),
                nn.Linear(hidden_size[i], hidden_size[i + 1]))

        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            self.classifier.add_module(
                'classifier_output',
                nn.Linear(hidden_size[len(hidden_size) - 1], 1))
        else:
            self.classifier.add_module(
                'classifier_output',
                nn.Linear(hidden_size[len(hidden_size) - 1], 2))

    def forward(self,
                task_type=None,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        if self.pool_method == 'first':
            pooled_output = outputs[0][:, 0]
        else:
            pooled_output = torch.mean(outputs[0], dim=1)

        # bert 输出和 task_type_embedding 相加
        task_type_embedding = self.task_type_embedding(task_type % 2)

#         sentence_w2v_embedding0 = self.w2v_linear(sentence_w2v_embedding0)
#         sentence_w2v_embedding1 = self.w2v_linear(sentence_w2v_embedding1)

        pooled_output = self.ln(pooled_output + task_type_embedding)
        pooled_output = self.dropout(pooled_output)
        out = self.classifier(pooled_output)

        # pooled_output = self.dropout(pooled_output)
        # out_list = []
        # for i in range(2):
        #     out = self.classifier_list[i](pooled_output)
        #     out_list.append(out)
        # out = torch.cat(out_list, 1)
        # output_weight = (task_type % 2).view(-1, 1)
        # out = out.gather(1, output_weight)

        loss = 0
        if labels is not None:
            loss = compute_loss(out, labels, loss_method=self.loss_method)

        if self.loss_method in ['binary', 'focal_loss', 'ghmc']:
            out = torch.sigmoid(out).flatten()

        return out, loss
