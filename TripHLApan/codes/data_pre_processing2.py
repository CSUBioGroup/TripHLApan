#coding=utf-8
import warnings
warnings.filterwarnings("ignore")

import re

# 辅助 get_data_from_file 函数
def load_independent_data_from_file(independent_data_file_path):
    independent_pep_seq_list_tmp = []
    independent_allele_seq_list_tmp = []

    independent_pep_seq_list = []
    independent_allele_seq_list = []


    in_f = open(independent_data_file_path, 'r')
    for line in in_f:
        cols = re.split('[\t\n]', line)
        independent_pep_seq_list_tmp.append(cols[0])
        independent_allele_seq_list_tmp.append(cols[1])

    in_f.close()


        # 裁剪pep
    for item in independent_pep_seq_list_tmp:
        peplen = len(item)
        insert_idx = int((peplen + 1) / 2) - 1
        pseq_pep_seq = item[0:insert_idx] + 'X' * (14 - peplen) + item[insert_idx:]
        independent_pep_seq_list.append(pseq_pep_seq)

        # 裁剪allele
    dict_allele_seq = map_allele_name_seq()
    for item in independent_allele_seq_list_tmp:
        allele_seq = dict_allele_seq[item]
        if len(allele_seq) >= 200:
            pseq_allele_seq = allele_seq[0:200]
        else:
            pseq_allele_seq = allele_seq + 'X' * (200 - len(allele_seq))
        independent_allele_seq_list.append(pseq_allele_seq)

    return independent_pep_seq_list, independent_allele_seq_list

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

