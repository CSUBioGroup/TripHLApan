import re
out_f = open('./pretrain_train.txt', 'w')
out_f2 = open('./pretrain_dev.txt', 'w')
in_f = open('./map_allele_seq.txt', 'r')
count = 0
for line in in_f:
    cols = re.split('[\t\n]', line)
    seq = cols[1]
    seq_sentence = ''
    count += 1
    for i in range(len(seq) - 1):
        seq_sentence = seq_sentence + seq[i] + ' '
    seq_sentence = seq_sentence + seq[-1]
    if count > 15000:
        out_f2.write(seq_sentence + '\n')
    else:
        out_f.write(seq_sentence + '\n')
in_f.close()
out_f2.close()
out_f.close()