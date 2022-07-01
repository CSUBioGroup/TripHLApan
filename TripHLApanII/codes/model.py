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

MODEL_SAVE_PATH = '../models/model/'
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

threshold = 0.2
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EMBED_SIZE = 6
K_Fold = 5

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
        self.gru3 = nn.GRU(20, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.full_conn1 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(True),
        )

        self.attention1 = nn.MultiheadAttention(256, 1)
        self.attention2 = nn.MultiheadAttention(256, 1)
        self.attention3 = nn.MultiheadAttention(256, 1)

        # network2
        self.embedding1 = nn.Embedding(21, 6)
        self.embedding2 = nn.Embedding(21, 6)
        self.embedding3 = nn.Embedding(21, 6)

        self.gru4 = nn.GRU(6, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru5 = nn.GRU(6, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru6 = nn.GRU(6, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.full_conn2 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(True),
        )

        self.attention4 = nn.MultiheadAttention(256, 1)
        self.attention5 = nn.MultiheadAttention(256, 1)
        self.attention6 = nn.MultiheadAttention(256, 1)

        # network3
        self.gru7 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru8 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.gru9 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True
                           )
        self.full_conn3 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(True),
        )

        self.attention7 = nn.MultiheadAttention(256, 1)
        self.attention8 = nn.MultiheadAttention(256, 1)
        self.attention9 = nn.MultiheadAttention(256, 1)

        # fully_conn
        self.full_conn = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )


    def forward(self, peps1, alleles11,alleles12, peps2, alleles21,alleles22, peps3, alleles31,alleles32):
        # network1
        x11 = self.gru1(peps1)[0]  # x1:(batch , 14,  256)
        query = x11.permute(1, 0, 2)
        key = x11.permute(1, 0, 2)
        value = x11.permute(1, 0, 2)
        x_attention1, __ = self.attention1(query, key, value)
        x11 = x_attention1.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x21 = self.gru2(alleles11)[0]  # x1:(batch , 200,  256)
        query = x21.permute(1, 0, 2)
        key = x21.permute(1, 0, 2)
        value = x21.permute(1, 0, 2)
        y_attention2, __ = self.attention2(query, key, value)
        x21 = y_attention2.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x31 = self.gru3(alleles12)[0]  # x1:(batch , 200,  256)
        query = x31.permute(1, 0, 2)
        key = x31.permute(1, 0, 2)
        value = x31.permute(1, 0, 2)
        y_attention3, __ = self.attention3(query, key, value)
        x31 = y_attention3.permute(1, 0, 2)[:, -1]  # x1:(batch , 256)

        x41 = torch.cat((x11, x21, x31), 1)  # x3:(batch , 768)
        result1 = self.full_conn1(x41)  # result:(batch, 128)

        # network2
        X1 = self.embedding1(peps2)  ## x1:(batch , 14,  6)
        Y1 = self.embedding2(alleles21)  ## x2:(batch , 200,  6)
        Y2 = self.embedding3(alleles22)  ## x2:(batch , 200,  6)

        x1 = self.gru4(X1)[0]  # x1:(batch , 14,  256)
        query = x1.permute(1, 0, 2)
        key = x1.permute(1, 0, 2)
        value = x1.permute(1, 0, 2)
        x_attention, __ = self.attention4(query, key, value)
        x1 = x_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x2 = self.gru5(Y1)[0]  # x1:(batch , 200,  256)
        query = x2.permute(1, 0, 2)
        key = x2.permute(1, 0, 2)
        value = x2.permute(1, 0, 2)
        y_attention, __ = self.attention5(query, key, value)
        x2 = y_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x3 = self.gru6(Y2)[0]  # x1:(batch , 200,  256)
        query = x3.permute(1, 0, 2)
        key = x3.permute(1, 0, 2)
        value = x3.permute(1, 0, 2)
        y_attention2, __ = self.attention6(query, key, value)
        x3 = y_attention2.permute(1, 0, 2)[:, -1]  # x1:(batch , 256)

        x4 = torch.cat((x1, x2, x3), 1)  # x3:(batch , 768)
        result2 = self.full_conn3(x4)  # result:(batch, 128)

        # network3
        x1 = self.gru7(peps3)[0]  # x1:(batch , 14,  256)
        query = x1.permute(1, 0, 2)
        key = x1.permute(1, 0, 2)
        value = x1.permute(1, 0, 2)
        x_attention, __ = self.attention7(query, key, value)
        x1 = x_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x2 = self.gru8(alleles31)[0]  # x1:(batch , 200,  256)
        query = x2.permute(1, 0, 2)
        key = x2.permute(1, 0, 2)
        value = x2.permute(1, 0, 2)
        y_attention, __ = self.attention8(query, key, value)
        x2 = y_attention.permute(1, 0, 2)[:,-1]  # x1:(batch , 256)

        x3 = self.gru9(alleles32)[0]  # x1:(batch , 200,  256)
        query = x3.permute(1, 0, 2)
        key = x3.permute(1, 0, 2)
        value = x3.permute(1, 0, 2)
        y_attention2, __ = self.attention9(query, key, value)
        x3 = y_attention2.permute(1, 0, 2)[:, -1]  # x1:(batch , 256)

        x4 = torch.cat((x1, x2, x3), 1)  # x3:(batch , 768)
        result3 = self.full_conn3(x4)  # result:(batch, 128)

        # fully_conn
        x = torch.cat((result1, result2, result3), 1)  # x3:(batch , 384)
        result = self.full_conn(x) # batch, 1
        return result
