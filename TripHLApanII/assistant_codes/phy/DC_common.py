# /usr/bin/python
# _*_ coding: utf-8 _*_
# desc: format protein chain file to get protein dipeptide composition s = 0
import os
import math

def getDC(chainfile, outfile,interval):
    with open(outfile, 'w') as fo:
        aa_1 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        aa_2 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        dipeptide_all = []
        for i in aa_1:
            for j in aa_2:
                dipeptide_all.append(i + j)
        DC = []
        with open(chainfile, 'r') as fr:
            flag = 0
            for eachline in fr:
                if flag % 2 == 0:
                    flag = flag+1
                    pdb_name = eachline.strip('\n')
                    DC.append(pdb_name+" ")
                else:
                    s = eachline
                    c = int(interval)
                    flag = flag + 1
                    for a in dipeptide_all:
                        i = 0
                        num = 0
                        while i < len(s) - 1 - c:
                            if s[i] == a[0]:
                                j = i + 1 + c
                                if s[j] == a[1]:
                                    num = num + 1
                            i = i + 1
                        #num = seq.count(sub,0,len(seq))
                        #oaac = float(num)/len(seq)
                        #oaac = round(math.sqrt(oaac),3)
			#print(oaac)
                        dc = float(num)/(len(s)-1)
                        dc = str(dc)
                        DC.append(dc + ' ')
                    DC.append('\n')
        #fo.write(''.join(sorted(ow)))
        fo.write(''.join(DC))
if __name__ == "__main__":
    chainfile = os.sys.argv[1]
    outfile = os.sys.argv[2]
    interval = os.sys.argv[3]
    getDC(chainfile, outfile, interval)
"""
python DC_common.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_single_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/DCSingle_0.data 0
"""
"""
python DC_common.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_single_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/DCSingle_1.data 1
"""
"""
python DC_common.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_single_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/DCSingle_2.data 2
"""





"""
python DC_common.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_double_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/DCDouble_0.data 0
"""
"""
python DC_common.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_double_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/DCDouble_1.data 1
"""
"""
python DC_common.py \
/ifs/gdata1/wangtong/non-redundantData/simplify_double_seq.fasta \
/ifs/gdata1/wangtong/non-redundantData/DCDouble_2.data 2
"""
