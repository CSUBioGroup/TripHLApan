import re

def get_mapping_directly():
    dict_allele_seq = {}

    allele_name_tmp = ''
    fasta_file = './hla_prot.fasta'
    in_f = open(fasta_file, 'r')
    for line in in_f:
        if '>' in line:
            cols = re.split('[ \n]', line)
            allele_name_tmp = cols[1]
            dict_allele_seq[allele_name_tmp] = ''
        else:
            dict_allele_seq[allele_name_tmp] += line.split()[0]
    in_f.close()

    return dict_allele_seq


def rectify_mapping_and_write_to_file(dict_allele_seq, out_path):
    dict_allele_seq_rectified = {}

    for allele in dict_allele_seq.keys():
        cols = re.split('[*:]', allele)
        if len(cols) == 1:
            rectified_allele_name = 'HLA-' + cols[0]
            if rectified_allele_name in dict_allele_seq_rectified.keys():
                continue
            else:
                dict_allele_seq_rectified[rectified_allele_name] = dict_allele_seq[allele]
        elif len(cols) == 2:
            rectified_allele_name = 'HLA-' + cols[0] + '*' + seq_retain_num(cols[1])
            if rectified_allele_name in dict_allele_seq_rectified.keys():
                continue
            else:
                dict_allele_seq_rectified[rectified_allele_name] = dict_allele_seq[allele]
        else:
            rectified_allele_name = 'HLA-' + cols[0] + '*' + seq_retain_num(cols[1]) + ':' + seq_retain_num(cols[2])
            if rectified_allele_name in dict_allele_seq_rectified.keys():
                continue
            else:
                dict_allele_seq_rectified[rectified_allele_name] = dict_allele_seq[allele]

    # 将得到的字典写入文件
    of = open(out_path, 'w')
    for allele in dict_allele_seq_rectified.keys():
        of.write(allele + '\t' + dict_allele_seq_rectified[allele] + '\n')
    of.close()


# 保留一个序列中的数字部分并返回
def seq_retain_num(seq):
    retain_seq = ''
    for ch in seq:
        if judge_num(ch):
            retain_seq += ch
            #print(ch)
        else:
            continue
    return retain_seq

# 判断一个字符是不是数字
def judge_num(ch):
    if ord(ch) >= ord('0') and ord(ch) <= ord('9'):
        is_num = True
    else:
        #print(ord(ch))
        is_num = False
    return is_num

out_path = './map_allele_seq.txt'
rectify_mapping_and_write_to_file(get_mapping_directly(), out_path)

