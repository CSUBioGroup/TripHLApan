#coding=utf-8
import warnings
warnings.filterwarnings("ignore")

import re

# 这一函数不需要修改，用于向model*.py发送数据输入前的处理的接口
# 简单从文件中去取出序列列表，处理后返回
def get_data_from_file(train_data_file_path, independent_data_file_path):
    # 从文件中读取数据，返回肽序列、allele序列、亲和力值和EL标签共4个list，无用list的为空
    #print('Q')
    pep_seq_list, allele1_seq_list,allele2_seq_list, EL_label_list = load_train_data_from_file(train_data_file_path)
    #print('W')
    # 从文件中读取数据，返回肽序列、allele序列、亲和力值和EL标签共4个list，无用list的为空
    independent_pep_seq_list, independent_allele1_seq_list, independent_allele2_seq_list, independent_EL_label_list = load_independent_data_from_file(independent_data_file_path)

    return pep_seq_list, allele1_seq_list,allele2_seq_list, EL_label_list, independent_pep_seq_list, independent_allele1_seq_list, independent_allele2_seq_list, independent_EL_label_list




# load_train_data_from_file 和 load_independent_data_from_file函数的作用：
# 加载文件，并且1、将allele名映射为对应的序列，2、将肽序列和allele序列进行变换并返回，
# 变换方式例如，截断后拼接等
# 注意4个列表数据按下标对应，注意序列尚未编码
# 辅助 get_data_from_file 函数
def load_train_data_from_file(train_data_file_path):
    pep_seq_list_tmp = []
    allele1_seq_list_tmp = []
    allele2_seq_list_tmp = []

    pep_seq_list = []
    allele1_seq_list = []
    allele2_seq_list = []
    EL_label_list = []
    #print('q')
    in_f = open(train_data_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        pep_seq_list_tmp.append(cols[0])
        allele1_seq_list_tmp.append(cols[1])
        allele2_seq_list_tmp.append(cols[2])
        EL_label_list.append(int(cols[3]))
    in_f.close()

    #print('w')
    # 裁剪pep
    for item in pep_seq_list_tmp:
        peplen = len(item)

        pseq_pep_seq = item + 'X' * (32 - peplen)
        pep_seq_list.append(pseq_pep_seq)
    #print('e')
    # 裁剪allele
    dict_allele_seq = map_allele_name_seq()
    for item in allele1_seq_list_tmp:
        start_inx = 0
        allele_seq = dict_allele_seq['HLA-' + item]
        if item[0:3] == 'DRA':
            start_inx = 25
        elif item[0:3] == 'DRB':
            start_inx = 29
        elif item[0:3] == 'DQA':
            start_inx = 23
        elif item[0:3] == 'DQB':
            start_inx = 32
        elif item[0:3] == 'DMA':
            start_inx = 26
        elif item[0:3] == 'DMB':
            start_inx = 18
        elif item[0:3] == 'DPA':
            start_inx = 28
        elif item[0:3] == 'DPB':
            start_inx = 29

        if len(allele_seq) > start_inx + 100:
            pseq_allele_seq = allele_seq[start_inx: start_inx + 100]
        elif len(allele_seq) > start_inx:
            pseq_allele_seq = allele_seq[start_inx: len(allele_seq)] + 'X' * (100 - len(allele_seq) + start_inx)
        else:
            pseq_allele_seq = 'X' * 100

        allele1_seq_list.append(pseq_allele_seq)
    for item in allele2_seq_list_tmp:
        start_inx = 0
        allele_seq = dict_allele_seq['HLA-' + item]
        if item[0:3] == 'DRA':
            start_inx = 25
        elif item[0:3] == 'DRB':
            start_inx = 29
        elif item[0:3] == 'DQA':
            start_inx = 23
        elif item[0:3] == 'DQB':
            start_inx = 32
        elif item[0:3] == 'DMA':
            start_inx = 26
        elif item[0:3] == 'DMB':
            start_inx = 18
        elif item[0:3] == 'DPA':
            start_inx = 28
        elif item[0:3] == 'DPB':
            start_inx = 29

        if len(allele_seq) > start_inx + 100:
            pseq_allele_seq = allele_seq[start_inx: start_inx + 100]
        elif len(allele_seq) > start_inx:
            pseq_allele_seq = allele_seq[start_inx: len(allele_seq)] + 'X' * (100 - len(allele_seq) + start_inx)
        else:
            pseq_allele_seq = 'X' * 100

        allele2_seq_list.append(pseq_allele_seq)

    return pep_seq_list, allele1_seq_list,allele2_seq_list, EL_label_list



# 辅助 get_data_from_file 函数
def load_independent_data_from_file(independent_data_file_path):
    pep_seq_list_tmp = []
    allele1_seq_list_tmp = []
    allele2_seq_list_tmp = []

    pep_seq_list = []
    allele1_seq_list = []
    allele2_seq_list = []
    EL_label_list = []
    #print('q')
    in_f = open(independent_data_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        pep_seq_list_tmp.append(cols[0])
        allele1_seq_list_tmp.append(cols[1])
        allele2_seq_list_tmp.append(cols[2])
        EL_label_list.append(int(cols[3]))
    in_f.close()

    #print('w')
    # 裁剪pep
    for item in pep_seq_list_tmp:
        peplen = len(item)

        pseq_pep_seq = item + 'X' * (32 - peplen)
        pep_seq_list.append(pseq_pep_seq)
    #print('e')
    # 裁剪allele
    dict_allele_seq = map_allele_name_seq()
    for item in allele1_seq_list_tmp:
        start_inx = 0
        allele_seq = dict_allele_seq['HLA-' + item]
        if item[0:3] == 'DRA':
            start_inx = 25
        elif item[0:3] == 'DRB':
            start_inx = 29
        elif item[0:3] == 'DQA':
            start_inx = 23
        elif item[0:3] == 'DQB':
            start_inx = 32
        elif item[0:3] == 'DMA':
            start_inx = 26
        elif item[0:3] == 'DMB':
            start_inx = 18
        elif item[0:3] == 'DPA':
            start_inx = 28
        elif item[0:3] == 'DPB':
            start_inx = 29

        if len(allele_seq) > start_inx + 100:
            pseq_allele_seq = allele_seq[start_inx: start_inx + 100]
        elif len(allele_seq) > start_inx:
            pseq_allele_seq = allele_seq[start_inx: len(allele_seq)] + 'X' * (100 - len(allele_seq) + start_inx)
        else:
            pseq_allele_seq = 'X' * 100

        allele1_seq_list.append(pseq_allele_seq)
    for item in allele2_seq_list_tmp:
        start_inx = 0
        allele_seq = dict_allele_seq['HLA-' + item]
        if item[0:3] == 'DRA':
            start_inx = 25
        elif item[0:3] == 'DRB':
            start_inx = 29
        elif item[0:3] == 'DQA':
            start_inx = 23
        elif item[0:3] == 'DQB':
            start_inx = 32
        elif item[0:3] == 'DMA':
            start_inx = 26
        elif item[0:3] == 'DMB':
            start_inx = 18
        elif item[0:3] == 'DPA':
            start_inx = 28
        elif item[0:3] == 'DPB':
            start_inx = 29

        if len(allele_seq) > start_inx + 100:
            pseq_allele_seq = allele_seq[start_inx: start_inx + 100]
        elif len(allele_seq) > start_inx:
            pseq_allele_seq = allele_seq[start_inx: len(allele_seq)] + 'X' * (100 - len(allele_seq) + start_inx)
        else:
            pseq_allele_seq = 'X' * 100

        allele2_seq_list.append(pseq_allele_seq)

    return pep_seq_list, allele1_seq_list,allele2_seq_list, EL_label_list


# 加载无标签文件
def load_independent_data_from_file_for_test(independent_data_file_path):
    pep_seq_list_tmp = []
    allele1_seq_list_tmp = []
    allele2_seq_list_tmp = []

    pep_seq_list = []
    allele1_seq_list = []
    allele2_seq_list = []
    #print('q')
    in_f = open(independent_data_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        pep_seq_list_tmp.append(cols[0])
        allele1_seq_list_tmp.append(cols[1])
        allele2_seq_list_tmp.append(cols[2])
    in_f.close()

    #print('w')
    # 裁剪pep
    for item in pep_seq_list_tmp:
        peplen = len(item)

        pseq_pep_seq = item + 'X' * (32 - peplen)
        pep_seq_list.append(pseq_pep_seq)
    #print('e')
    # 裁剪allele
    dict_allele_seq = map_allele_name_seq()
    for item in allele1_seq_list_tmp:
        start_inx = 0
        allele_seq = dict_allele_seq['HLA-' + item]
        if item[0:3] == 'DRA':
            start_inx = 25
        elif item[0:3] == 'DRB':
            start_inx = 29
        elif item[0:3] == 'DQA':
            start_inx = 23
        elif item[0:3] == 'DQB':
            start_inx = 32
        elif item[0:3] == 'DMA':
            start_inx = 26
        elif item[0:3] == 'DMB':
            start_inx = 18
        elif item[0:3] == 'DPA':
            start_inx = 28
        elif item[0:3] == 'DPB':
            start_inx = 29

        if len(allele_seq) > start_inx + 100:
            pseq_allele_seq = allele_seq[start_inx: start_inx + 100]
        elif len(allele_seq) > start_inx:
            pseq_allele_seq = allele_seq[start_inx: len(allele_seq)] + 'X' * (100 - len(allele_seq) + start_inx)
        else:
            pseq_allele_seq = 'X' * 100

        allele1_seq_list.append(pseq_allele_seq)
    for item in allele2_seq_list_tmp:
        start_inx = 0
        allele_seq = dict_allele_seq['HLA-' + item]
        if item[0:3] == 'DRA':
            start_inx = 25
        elif item[0:3] == 'DRB':
            start_inx = 29
        elif item[0:3] == 'DQA':
            start_inx = 23
        elif item[0:3] == 'DQB':
            start_inx = 32
        elif item[0:3] == 'DMA':
            start_inx = 26
        elif item[0:3] == 'DMB':
            start_inx = 18
        elif item[0:3] == 'DPA':
            start_inx = 28
        elif item[0:3] == 'DPB':
            start_inx = 29

        if len(allele_seq) > start_inx + 100:
            pseq_allele_seq = allele_seq[start_inx: start_inx + 100]
        elif len(allele_seq) > start_inx:
            pseq_allele_seq = allele_seq[start_inx: len(allele_seq)] + 'X' * (100 - len(allele_seq) + start_inx)
        else:
            pseq_allele_seq = 'X' * 100

        allele2_seq_list.append(pseq_allele_seq)

    return pep_seq_list, allele1_seq_list,allele2_seq_list
    

# 加载allele与其序列的映射文件，返回映射字典
def map_allele_name_seq():
    dict_allele_seq = {}
    map_allele_seq_file_path = '../assistant_codes/map_allele_seq.txt'
    in_f = open(map_allele_seq_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        dict_allele_seq[cols[0]] = cols[1]
    in_f.close()
    return dict_allele_seq

#
# train_data_file_path = '../data/tt.txt'
# pep_seq_list, allele1_seq_list,allele2_seq_list, EL_label_list = load_train_data_from_file(train_data_file_path)
# print('pep_seq_list', pep_seq_list)
# print('allele1_seq_list', allele1_seq_list)
# print('allele2_seq_list', allele2_seq_list)
# print('EL_label_list', EL_label_list)
