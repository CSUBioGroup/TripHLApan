#coding=utf-8
import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from helper import *

# Configuration
#region Configuration
# need Constant revision
from data_pre_processing2 import *

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random_seed = 0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)


# 训练数据和独立测试数据的来源文件夹，注意传入前先将数据进行格式化，列分别为pep，allele，affinity值，EL标签，共4列

threshold = 0.5
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
EMBED_SIZE = 6
K_Fold = 5

# one-hot/BLOSUM50/BLOSUM62/embedding/num/AAfea_phy，
# 注意如果选择embedding参数转移到这里进行序列编码，embedding不能写入网络模型里面，也就是embedding的参数不能再学习
# 如果直接在模型中用embedding，那么在输入模型之前不应该编码，只需要简单讲字母改为数字即可，在MyDataSet_Long中将数据转为LongTensor类型


# 用flag控制使用的数据标签类型，例：选择flag = BA，则输出EL_label_list为空
flag_label = 'EL' # BA/EL/both

#region Configuration


class Network_conn(nn.Module):
    def __init__(self):
        super(Network_conn, self).__init__()

        # network1
        self.gru1 = nn.GRU(20, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru2 = nn.GRU(20, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.full_conn1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        # network2
        self.embedding1 = nn.Embedding(21, 6)
        self.embedding2 = nn.Embedding(21, 6)

        self.gru3 = nn.GRU(6, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru4 = nn.GRU(6, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.full_conn2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        # network3
        self.gru5 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru6 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.full_conn3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        self.attention1 = nn.MultiheadAttention(256, 1)
        self.attention2 = nn.MultiheadAttention(256, 1)
        self.attention3 = nn.MultiheadAttention(256, 1)
        self.attention4 = nn.MultiheadAttention(256, 1)
        self.attention5 = nn.MultiheadAttention(256, 1)
        self.attention6 = nn.MultiheadAttention(256, 1)

        # fully_conn
        self.full_conn = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )


    def forward(self, peps1, alleles1, peps2, alleles2, peps3, alleles3):
        # network1
        x1 = self.gru1(peps1)[0]  # x1:(batch , 14,  256)
        query = x1.permute(1, 0, 2)
        key = x1.permute(1, 0, 2)
        value = x1.permute(1, 0, 2)
        x_attention, __ = self.attention1(query, key, value)
        x1 = x_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x2 = self.gru2(alleles1)[0]  # x1:(batch , 200,  256)
        query = x2.permute(1, 0, 2)
        key = x2.permute(1, 0, 2)
        value = x2.permute(1, 0, 2)
        y_attention, __ = self.attention2(query, key, value)
        x2 = y_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x3 = torch.cat((x1, x2), 1)  # x3:(batch , 512)
        result1 = self.full_conn1(x3)  # result1:(batch, 128)

        #network2
        X1 = self.embedding1(peps2)  ## x1:(batch , 14,  6)
        Y1 = self.embedding2(alleles2)  ## x2:(batch , 200,  6)

        x1 = self.gru3(X1)[0]  # x1:(batch , 14,  256)
        query = x1.permute(1, 0, 2)
        key = x1.permute(1, 0, 2)
        value = x1.permute(1, 0, 2)
        x_attention, __ = self.attention3(query, key, value)
        x1 = x_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x2 = self.gru4(Y1)[0]  # x1:(batch , 200,  256)
        query = x2.permute(1, 0, 2)
        key = x2.permute(1, 0, 2)
        value = x2.permute(1, 0, 2)
        y_attention, __ = self.attention4(query, key, value)
        x2 = y_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x3 = torch.cat((x1, x2), 1)  # x3:(batch , 512)
        result2 = self.full_conn2(x3)  # result:(batch, 128)

        # network3
        x1 = self.gru5(peps3)[0]  # x1:(batch , 14,  256)
        query = x1.permute(1, 0, 2)
        key = x1.permute(1, 0, 2)
        value = x1.permute(1, 0, 2)
        x_attention, __ = self.attention5(query, key, value)
        x1 = x_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x2 = self.gru6(alleles3)[0]  # x1:(batch , 200,  256)
        query = x2.permute(1, 0, 2)
        key = x2.permute(1, 0, 2)
        value = x2.permute(1, 0, 2)
        y_attention, __ = self.attention6(query, key, value)
        x2 = y_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x3 = torch.cat((x1, x2), 1)  # x3:(batch , 512)
        result3 = self.full_conn3(x3)  # result:(batch, 128)

        # fully_conn
        x = torch.cat((result1, result2, result3), 1)  # x3:(batch , 384)
        result = self.full_conn(x) # batch, 1
        return result
