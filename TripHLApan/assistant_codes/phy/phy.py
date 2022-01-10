#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
根据fasta文件进行编码phy特征
'''

import os

def encode_phy_by_fasta(fasta, AAfea_phy, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fr_fasta = open(fasta,'r')
    fo_fea = open(AAfea_phy, 'r')
    fr_fea = fo_fea.readlines()
    chainname = []
    for eachline in fr_fasta:
        if '>' in eachline:

            #chainname = eachline[1:7]
            chainname = eachline[1:6]#独立测试集专用

            outname = chainname + '.data'
            #print(chainname)
            #print(outname)

            fw_feat = open(outdir + '/' + outname, 'w')
            continue
        seq = eachline.strip()
        for i in range(len(seq)):
            content = []
            AA_code = seq[i]
            for onelinefea in fr_fea:
                linefea = onelinefea.split('\t')
                #print(linefea)
                if AA_code == linefea[1].strip():
                    for j in range(len(linefea) - 2):
                        j = j + 1
                        #print(linefea[j+1])
                        content.append(linefea[j+1].strip() + '\t') #遍历属性
                    #print(content)
                    break
            fw_feat.write(''.join(content) + '\n')
        fw_feat.close()
    fr_fasta.close()

if __name__=="__main__":
    params1 = os.sys.argv[1]
    params2 = os.sys.argv[2]
    params3 = os.sys.argv[3]
    #encode_phy(params1, params2, params3)
    encode_phy_by_fasta(params1, params2, params3)

'''
Linux Command:


# 方法二：fasta
nohup python -u phy.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_single_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/AAIndex_phy/AAfea_phy.txt \
/ifs/gdata1/wangtong/non-redundantData/AAIndex_phy/phy_single \
> phy_single.out 2>&1 &

nohup python -u phy.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_double_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/AAIndex_phy/AAfea_phy.txt \
/ifs/gdata1/wangtong/non-redundantData/AAIndex_phy/phy_double \
> phy_double.out 2>&1 &

python phy.py \
/ifs/gdata1/wangtong/non-redundantData/AAIndex_phy/test.fa \
/ifs/gdata1/wangtong/non-redundantData/AAIndex_phy/AAfea_phy.txt \
/ifs/gdata1/wangtong/non-redundantData/AAIndex_phy/phy_test

nohup python -u encode_phy.py \
/ifs/home/liudiwei/PredRBR/data/RBP86/base_data/RBP86_pseq.fasta \
/ifs/home/liudiwei/PredRBR/data/_Global_data/AAfea_phy.txt \
/ifs/home/liudiwei/PredRBR/data/RBP86/encode/basefeat/encode_phy2010 \
> encode_phy_86.out 2>&1 &
'''